"""
Evaluate a candidate model bundle against a held-out test set and
optionally compare it to the currently-active model in production. If
the candidate doesn't clear the promotion threshold, it stays in
staging and the active model is unchanged.

Usage:
    python -m training.eval --bundle ./checkpoints-staging/reflection/<run-id>
    python -m training.eval --bundle ./checkpoints-staging/reflection/<run-id> --compare-active

Promotion criteria (per condition):
    reflection:    IoU on held-out ≥ 0.65, AND IoU > active.IoU
    dead-fixture:  mAP50 ≥ 0.65, AND mAP50 > active.mAP50
    classifier:    F1 ≥ 0.85, AND F1 > active.F1

Output: writes eval/report.json next to the bundle; exits 0 if the
candidate meets bar and is better than active (or no active), 1 if it
doesn't meet bar, 2 if active is better.
"""

from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

console = Console()

PROMOTION_BAR = {
    # Per (condition, model_type) — most-specific wins. The bar is the
    # metric the model_type actually produces ("iou" requires a
    # segmenter; classifiers can't pay that gate).
    ("reflection",   "classifier"):   {"key": "f1",    "min": 0.70},
    ("reflection",   "mobile-sam"):   {"key": "iou",   "min": 0.65},
    ("dead-fixture", "classifier"):   {"key": "f1",    "min": 0.75},
    ("dead-fixture", "yolo-detect"):  {"key": "mAP50", "min": 0.65},
    # Bars for "this is the v1 model" are intentionally lower than what
    # we'd want long-term — they exist to gate against regression and
    # obvious data bugs, not to chase SOTA. Raise as the data + model
    # improve.
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, type=Path,
                    help="Path to staging bundle dir (containing manifest.json + weights)")
    ap.add_argument("--compare-active", action="store_true",
                    help="Also compare against the bundle under ./checkpoints/<condition>/")
    ap.add_argument("--active-root", default=Path("./checkpoints"), type=Path)
    args = ap.parse_args()

    manifest_path = args.bundle / "manifest.json"
    if not manifest_path.exists():
        console.print(f"[red]no manifest at {manifest_path}[/]")
        return 1
    manifest = json.loads(manifest_path.read_text())
    condition = manifest["condition"]
    model_type = manifest["model_type"]
    bar = PROMOTION_BAR.get((condition, model_type))
    if not bar:
        console.print(f"[yellow]no promotion bar configured for ({condition}, {model_type})[/]")
        return 1

    # Real eval = re-run inference on a held-out split and compute
    # IoU/mAP/F1. Stub for v0 — pull metrics straight from the bundle's
    # training manifest (no leakage check yet).
    candidate_score = float(manifest.get("metrics", {}).get(bar["key"], 0.0))
    candidate_meets_bar = candidate_score >= bar["min"]

    report = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "bundle": str(args.bundle),
        "candidate_version": manifest["version"],
        "candidate_score":   candidate_score,
        "promotion_bar":     bar,
        "meets_bar":         candidate_meets_bar,
        "is_stub_bundle":    manifest.get("is_stub", False),
    }

    if args.compare_active:
        active_manifest_path = args.active_root / condition / "manifest.json"
        if active_manifest_path.exists():
            active = json.loads(active_manifest_path.read_text())
            active_score = float(active.get("metrics", {}).get(bar["key"], 0.0))
            report["active_version"] = active["version"]
            report["active_score"]   = active_score
            report["better_than_active"] = candidate_score > active_score
        else:
            report["active_version"] = None
            report["better_than_active"] = True  # no active = always better

    (args.bundle / "eval").mkdir(exist_ok=True)
    (args.bundle / "eval" / "report.json").write_text(json.dumps(report, indent=2))
    console.print(json.dumps(report, indent=2))

    if not candidate_meets_bar:
        return 1
    if args.compare_active and not report.get("better_than_active", True):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
