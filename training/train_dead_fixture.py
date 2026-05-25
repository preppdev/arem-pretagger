"""
Train (or fine-tune) the dead-fixture detector.

Architecture: two-stage.
  Stage 1: YOLOv8n detector that locates light fixtures in any room.
           Trains on bbox labels — initially bootstrapped by running
           Gemini once on the verified positives to generate boxes,
           then refined by manual correction in /verify-labels (a
           future bbox-mode addition).
  Stage 2: small classifier head that takes the cropped fixture and
           predicts on/off. Trains on the same verified positives
           (cropped) + verified negatives' fixtures (positives by
           detector, no-correction-needed by reviewer).

For v0 (this commit), only the detector half is scaffolded — the
classification stage is added once the detector produces clean crops.

Usage:
    python -m training.train_dead_fixture \\
        --manifest ./data/dead-fixture/<run-id>/manifest.json \\
        --epochs 50
"""

from __future__ import annotations
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

console = Console()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out-root", default=Path("./checkpoints-staging"), type=Path)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=0.4)
    args = ap.parse_args()

    if not args.manifest.exists():
        console.print(f"[red]manifest not found: {args.manifest}[/]")
        return 1
    manifest = json.loads(args.manifest.read_text())
    if manifest["condition"] != "dead-fixture":
        console.print(f"[red]manifest is for {manifest['condition']}, not dead-fixture[/]")
        return 1

    pos_rows = [r for r in manifest["rows"] if r["verified"]]
    neg_rows = [r for r in manifest["rows"] if not r["verified"]]
    console.print(f"verified positives: {len(pos_rows)} (each contains ≥1 dead fixture)")
    console.print(f"verified negatives: {len(neg_rows)} (all fixtures correctly lit / no fixtures)")

    # Real training will use ultralytics:
    #   from ultralytics import YOLO
    #   model = YOLO("yolov8n.pt")
    #   results = model.train(data=yaml_config, epochs=args.epochs, ...)
    #   model.export(format="onnx" or "torchscript")
    #
    # Bbox labels come from a one-time bootstrap pass that runs Gemini
    # on each verified-positive image with a prompt like "return the
    # bounding box of every off light fixture as JSON [x,y,w,h]". The
    # boxes are then loaded into a future /verify-bboxes UI for human
    # correction before training.
    console.print("[yellow]_stub_train: detector half placeholder — see comments[/]")

    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    version = f"dead-fixture-yolov8n-v0-{run_id}"
    out_dir = args.out_root / "dead-fixture" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "weights.pt").write_bytes(b"stub-weights-replace-on-real-training")

    manifest_sha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()[:16]
    bundle_manifest = {
        "schema_version": 1,
        "condition": "dead-fixture",
        "model_type": "yolo-detect",
        "version": version,
        "weights": "weights.pt",
        "threshold": args.threshold,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "input_manifest_sha": manifest_sha,
        "hyperparameters": {
            "val_frac": args.val_frac, "epochs": args.epochs, "seed": args.seed,
        },
        "split": {
            "positives": len(pos_rows),
            "negatives": len(neg_rows),
        },
        "metrics": {"stub": True, "mAP50": 0.0, "precision": 0.0, "recall": 0.0},
        "is_stub": True,
    }
    (out_dir / "manifest.json").write_text(json.dumps(bundle_manifest, indent=2))

    console.print(f"\n[green]wrote bundle:[/] {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
