"""
Train (or fine-tune) the reflection segmenter on a manifest produced
by export_dataset.py.

Approach: MobileSAM ViT-Tiny backbone + a thin trainable decoder.
Mostly we want to leverage SAM's general "what is salient here"
knowledge and just teach the decoder to call reflections specifically.
This is the standard low-data fine-tune pattern.

Output:
    <out-dir>/<run-id>/
        weights.pt          # state_dict of the decoder
        manifest.json       # input manifest sha + hyperparams + metrics
        eval/               # per-image IoU on val split
        loss_curve.png      # tensorboard not required — single plot
        train.log

Usage:
    python -m training.train_reflection \\
        --manifest ./data/reflection/2026-05-28T18-42-00Z/manifest.json \\
        --val-frac 0.2 \\
        --epochs 30
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path,
                    help="Path to manifest.json from export_dataset.py")
    ap.add_argument("--out-root", default=Path("./checkpoints-staging"), type=Path,
                    help="Where to write trained weights (staging — promote separately)")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Sigmoid threshold for mask binarization at inference time")
    args = ap.parse_args()

    if not args.manifest.exists():
        console.print(f"[red]manifest not found: {args.manifest}[/]")
        return 1
    manifest = json.loads(args.manifest.read_text())
    if manifest["condition"] != "reflection":
        console.print(f"[red]manifest is for {manifest['condition']}, not reflection[/]")
        return 1

    # Rows with both an image and a mask are trainable for segmentation.
    # Confirmed-negative rows are used as background-only batches.
    pos_rows = [r for r in manifest["rows"] if r["verified"] and r["mask_path"]]
    neg_rows = [r for r in manifest["rows"] if not r["verified"]]
    console.print(f"trainable positives (mask present): {len(pos_rows)}")
    console.print(f"confirmed negatives:                {len(neg_rows)}")
    if len(pos_rows) < 50:
        console.print("[yellow]warning: fewer than 50 positives — model will undertrain[/]")

    # Deterministic split
    random.seed(args.seed)
    random.shuffle(pos_rows)
    split_at = int(len(pos_rows) * (1.0 - args.val_frac))
    train_pos, val_pos = pos_rows[:split_at], pos_rows[split_at:]
    console.print(f"train/val: {len(train_pos)} / {len(val_pos)}")

    # ----- TRAINING -----
    #
    # The real loop lives below this comment when the deps are in place.
    # Until ~250 verified positives are in hand, this stub just persists
    # a placeholder bundle so the inference service can be plumbed end-
    # to-end. Replace _stub_train() with the MobileSAM fine-tune loop
    # when the data is ready.
    #
    # Reference: https://github.com/ChaoningZhang/MobileSAM
    #   from mobile_sam import sam_model_registry, SamPredictor
    #   sam = sam_model_registry["vit_t"](checkpoint="mobile_sam.pt").cuda()
    #   # freeze backbone; train only mask_decoder
    #   for p in sam.image_encoder.parameters(): p.requires_grad = False
    #   for p in sam.prompt_encoder.parameters(): p.requires_grad = False
    #   opt = torch.optim.AdamW(sam.mask_decoder.parameters(), lr=args.lr)
    #   ...
    #
    metrics = _stub_train(train_pos, val_pos, args)

    # ----- WRITE BUNDLE -----
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    version = f"reflection-mobilesam-v0-{run_id}"
    out_dir = args.out_root / "reflection" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Placeholder weights file so the registry has something to point at.
    # Real training overwrites this with actual state_dict bytes.
    (out_dir / "weights.pt").write_bytes(b"stub-weights-replace-on-real-training")

    manifest_sha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()[:16]
    bundle_manifest = {
        "schema_version": 1,
        "condition": "reflection",
        "model_type": "mobile-sam",
        "version": version,
        "weights": "weights.pt",
        "threshold": args.threshold,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "input_manifest_sha": manifest_sha,
        "input_manifest_path": str(args.manifest),
        "hyperparameters": {
            "val_frac": args.val_frac,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "split": {
            "train_size": len(train_pos),
            "val_size":   len(val_pos),
            "negatives":  len(neg_rows),
        },
        "metrics": metrics,
        "is_stub": True,  # FLIP TO False ONCE _stub_train IS REPLACED
    }
    (out_dir / "manifest.json").write_text(json.dumps(bundle_manifest, indent=2))

    console.print(f"\n[green]wrote bundle:[/] {out_dir}")
    console.print("  - weights.pt (STUB — replace _stub_train with real loop)")
    console.print("  - manifest.json")
    console.print("\nNext steps:")
    console.print("  1. Replace _stub_train() with the MobileSAM fine-tune loop")
    console.print("  2. Run training (~10-30 min on 3090 for ~250 examples)")
    console.print("  3. python -m training.eval --bundle <this-dir>")
    console.print("  4. python scripts/promote_checkpoint.py reflection <this-dir>")
    return 0


def _stub_train(train_pos: list[dict[str, Any]], val_pos: list[dict[str, Any]], args) -> dict[str, Any]:
    """Placeholder for the real training loop. Returns dummy metrics so
    the bundle has the right shape; the inference path still falls
    through the model_registry stubs which return zero confidence."""
    console.print("[yellow]_stub_train: placeholder — see comments above for real loop[/]")
    return {
        "stub": True,
        "iou": 0.0,
        "f1": 0.0,
        "val_size": len(val_pos),
        "train_size": len(train_pos),
    }


if __name__ == "__main__":
    sys.exit(main())
