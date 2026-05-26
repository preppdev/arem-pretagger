"""
Train a binary classifier for "is there a reflection in this image?" on
the verified-positive + verified-negative ConditionLabel rows.

This is the v1 pretagger model for reflection — outputs a confidence
score, no mask. Reviewer UI uses the score to surface suggestions; the
existing /api/enhancement flow handles repair via Gemini-with-bbox.

Mask-based segmentation is a future v2 — needs SAM masks bootstrapped
on the same image cohort first.

Architecture:
  - ResNet-18 pretrained on ImageNet (in torchvision)
  - Frozen backbone (no point retraining ImageNet features on 500 images)
  - New 2-layer head: fc -> ReLU -> Dropout -> sigmoid logit
  - Binary cross-entropy loss
  - 80/20 train/val split, deterministic by seed
  - Augmentations: random horizontal flip, mild color jitter, random crop
  - AdamW, lr 1e-3 on head + 1e-5 on backbone (when unfrozen for fine-tune)
  - Early stop on val F1

Output:
  <out-root>/<run-id>/
    weights.pth          # state_dict for the model
    manifest.json        # version, hyperparams, metrics, model_type=classifier
    eval/                # per-image scores + confusion matrix

Usage:
  python -m training.train_reflection --manifest ./data/reflection/<run>/manifest.json
"""

from __future__ import annotations
import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from rich.console import Console
from rich.progress import track
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

console = Console()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReflectionDataset(Dataset):
    """One-image-per-row dataset. The manifest row tells us where the
    image lives on disk + whether it's a positive."""

    def __init__(self, rows: list[dict[str, Any]], image_root: Path, transform):
        self.rows = rows
        self.image_root = image_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        img_path = self.image_root / row["image_path"]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(1.0 if row["verified"] else 0.0, dtype=torch.float32)
        return img, label


def build_model() -> nn.Module:
    backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # Freeze everything except the final fc
    for p in backbone.parameters():
        p.requires_grad = False
    # Replace fc with a 2-layer head
    in_features = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, 1),  # binary logit
    )
    return backbone


def make_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def evaluate(model, loader, threshold: float = 0.5) -> dict[str, float]:
    model.eval()
    tp = fp = tn = fn = 0
    all_scores: list[tuple[float, int]] = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            logits = model(imgs).squeeze(-1).cpu()
            probs = torch.sigmoid(logits)
            for p, y in zip(probs.tolist(), labels.tolist()):
                all_scores.append((p, int(y)))
                pred = 1 if p >= threshold else 0
                if pred == 1 and y == 1: tp += 1
                elif pred == 1 and y == 0: fp += 1
                elif pred == 0 and y == 0: tn += 1
                else: fn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "scores": all_scores,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out-root", default=Path("./checkpoints-staging"), type=Path)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    if not args.manifest.exists():
        console.print(f"[red]manifest not found: {args.manifest}[/]")
        return 1
    manifest = json.loads(args.manifest.read_text())
    if manifest["condition"] != "reflection":
        console.print(f"[red]manifest is for {manifest['condition']}, not reflection[/]")
        return 1

    # Use BOTH positives and negatives — classifier needs both.
    rows = [r for r in manifest["rows"] if r.get("image_path")]
    rows = [r for r in rows if r.get("source") != "image-unavailable"]
    pos = [r for r in rows if r["verified"]]
    neg = [r for r in rows if not r["verified"]]
    console.print(f"trainable: [green]{len(pos)} positives[/] + [red]{len(neg)} negatives[/]")
    if len(pos) < 50 or len(neg) < 50:
        console.print("[yellow]warning: under 50 of one class — model will be unreliable[/]")

    random.seed(args.seed)
    random.shuffle(pos)
    random.shuffle(neg)
    split_p = int(len(pos) * (1.0 - args.val_frac))
    split_n = int(len(neg) * (1.0 - args.val_frac))
    train_rows = pos[:split_p] + neg[:split_n]
    val_rows   = pos[split_p:] + neg[split_n:]
    random.shuffle(train_rows)
    random.shuffle(val_rows)
    console.print(f"train/val: {len(train_rows)} / {len(val_rows)}")

    image_root = args.manifest.parent
    train_ds = ReflectionDataset(train_rows, image_root, make_transforms(train=True))
    val_ds   = ReflectionDataset(val_rows,   image_root, make_transforms(train=False))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    model = build_model().to(DEVICE)
    # Only the head's params have requires_grad=True after build_model
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=args.lr_head, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    best_f1 = 0.0
    best_metrics: dict[str, float] = {}
    best_state: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, labels in track(train_loader, description=f"epoch {epoch}/{args.epochs}"):
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            logits = model(imgs).squeeze(-1)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        avg_loss = running_loss / len(train_ds)
        val_metrics = evaluate(model, val_loader, args.threshold)
        history.append({
            "epoch": epoch, "train_loss": avg_loss,
            "val_f1": val_metrics["f1"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_accuracy": val_metrics["accuracy"],
        })
        console.print(
            f"  epoch {epoch:>2}  loss={avg_loss:.4f}  "
            f"val_f1={val_metrics['f1']:.3f}  P={val_metrics['precision']:.3f}  R={val_metrics['recall']:.3f}  "
            f"acc={val_metrics['accuracy']:.3f}"
        )
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_metrics = {k: v for k, v in val_metrics.items() if k != "scores"}
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        console.print("[red]no improvement across any epoch — keeping last state[/]")
        best_state = model.state_dict()
        best_metrics = {k: v for k, v in evaluate(model, val_loader, args.threshold).items() if k != "scores"}

    # ----- Write bundle -----
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    version = f"reflection-resnet18-v1-{run_id}"
    out_dir = args.out_root / "reflection" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / "weights.pth")

    manifest_sha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()[:16]
    bundle_manifest = {
        "schema_version": 1,
        "condition": "reflection",
        "model_type": "classifier",
        "version": version,
        "weights": "weights.pth",
        "threshold": args.threshold,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "input_manifest_sha": manifest_sha,
        "input_manifest_path": str(args.manifest),
        "hyperparameters": {
            "val_frac": args.val_frac, "epochs": args.epochs,
            "lr_head": args.lr_head, "batch_size": args.batch_size,
            "seed": args.seed, "backbone": "resnet18", "frozen_backbone": True,
        },
        "split": {
            "train_pos": len([r for r in train_rows if r["verified"]]),
            "train_neg": len([r for r in train_rows if not r["verified"]]),
            "val_pos":   len([r for r in val_rows if r["verified"]]),
            "val_neg":   len([r for r in val_rows if not r["verified"]]),
        },
        "metrics": best_metrics,
        "history": history,
        "is_stub": False,
    }
    (out_dir / "manifest.json").write_text(json.dumps(bundle_manifest, indent=2))

    # Eval artifacts
    (out_dir / "eval").mkdir(exist_ok=True)
    final_eval = evaluate(model, val_loader, args.threshold)
    (out_dir / "eval" / "val_scores.json").write_text(json.dumps({
        "threshold": args.threshold,
        "metrics": {k: v for k, v in final_eval.items() if k != "scores"},
        "per_image": final_eval["scores"],
    }, indent=2))

    console.print(f"\n[green]bundle written:[/] {out_dir}")
    console.print(f"  best F1: {best_f1:.3f}  (precision {best_metrics['precision']:.3f}, recall {best_metrics['recall']:.3f})")
    console.print(f"\nNext: python -m training.eval --bundle {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
