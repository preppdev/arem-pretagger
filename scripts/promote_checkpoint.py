"""
Promote a staged training bundle to "active" — copies it locally over
the current active bundle AND uploads it to R2 so other boxes can pull
via sync_checkpoints.sh.

Workflow:
    1. python -m training.train_reflection --manifest ...
       → writes ./checkpoints-staging/reflection/<run-id>/
    2. python -m training.eval --bundle ./checkpoints-staging/reflection/<run-id>
       → eval/report.json, exit 0 if it clears the bar
    3. python scripts/promote_checkpoint.py reflection \\
         ./checkpoints-staging/reflection/<run-id>
       → ./checkpoints/reflection/{manifest.json, weights.*}
       → R2: arem-training-data/pretagger-models/reflection/active/{manifest.json, weights.*}
    4. curl -X POST -H "X-Pretag-Token: $PRETAG_TOKEN" \\
            http://localhost:8090/reload
       → service hot-reloads the registry without restart

The previous active bundle is rotated to:
    arem-training-data/pretagger-models/<condition>/previous/
so a one-step rollback is `cp previous → active` then /reload.
"""

from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import boto3
from rich.console import Console

console = Console()

R2_BUCKET = "arem-training-data"
R2_PREFIX = "pretagger-models"


def _r2():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("condition", help="reflection | dead-fixture | ...")
    ap.add_argument("bundle_dir", type=Path, help="Path to the staged bundle")
    ap.add_argument("--active-root", default=Path("./checkpoints"), type=Path)
    ap.add_argument("--no-r2", action="store_true",
                    help="Skip the R2 upload (local-only promote)")
    args = ap.parse_args()

    if not args.bundle_dir.exists():
        console.print(f"[red]bundle not found: {args.bundle_dir}[/]")
        return 1
    manifest_path = args.bundle_dir / "manifest.json"
    if not manifest_path.exists():
        console.print(f"[red]no manifest.json in bundle[/]")
        return 1
    manifest = json.loads(manifest_path.read_text())
    if manifest["condition"] != args.condition:
        console.print(f"[red]bundle is for {manifest['condition']}, not {args.condition}[/]")
        return 1
    weights_name = manifest["weights"]
    weights_path = args.bundle_dir / weights_name
    if not weights_path.exists():
        console.print(f"[red]weights file missing: {weights_path}[/]")
        return 1

    # Local copy: ./checkpoints/<condition>/{manifest.json, weights.*}
    active_dir = args.active_root / args.condition
    active_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, active_dir / "manifest.json")
    shutil.copy2(weights_path,  active_dir / weights_name)
    console.print(f"[green]local active updated:[/] {active_dir}")

    if args.no_r2:
        console.print("[yellow]--no-r2: skipping R2 upload[/]")
        return 0

    r2 = _r2()
    # Rotate current active → previous (so rollback = swap)
    active_prefix   = f"{R2_PREFIX}/{args.condition}/active"
    previous_prefix = f"{R2_PREFIX}/{args.condition}/previous"
    try:
        listed = r2.list_objects_v2(Bucket=R2_BUCKET, Prefix=f"{active_prefix}/")
        for obj in listed.get("Contents", []) or []:
            old_key = obj["Key"]
            new_key = old_key.replace(active_prefix, previous_prefix, 1)
            r2.copy_object(Bucket=R2_BUCKET, CopySource={"Bucket": R2_BUCKET, "Key": old_key}, Key=new_key)
            r2.delete_object(Bucket=R2_BUCKET, Key=old_key)
        console.print(f"  rotated {len(listed.get('Contents', []) or [])} active → previous")
    except Exception as e:
        console.print(f"  [yellow]rotation skipped: {e}[/]")

    # Upload new active
    for local in [manifest_path, weights_path]:
        key = f"{active_prefix}/{local.name}"
        r2.upload_file(str(local), R2_BUCKET, key)
        console.print(f"  uploaded {key}")

    console.print(f"\n[green]promoted {args.condition} → {manifest['version']}[/]")
    console.print("Reload the service:")
    console.print(f'  curl -X POST -H "X-Pretag-Token: $PRETAG_TOKEN" http://localhost:8090/reload')
    return 0


if __name__ == "__main__":
    sys.exit(main())
