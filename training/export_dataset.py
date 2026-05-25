"""
Export the verified ConditionLabel rows + paired images/masks into a
versioned local training manifest. Run on the 3090 box; reads the same
DATABASE_URL the dashboard uses and pulls images from R2.

Output layout (under --out-dir, default ./data/<condition>/<run-id>/):

    <out-dir>/
        manifest.json          # full per-row provenance + label
        images/<sha>.jpg       # deduped by sha256 of bytes
        masks/<sha>.png        # reflection-only, if available

The manifest is the source of truth — paths, labels, source attribution,
the (imageReviewId, condition) tuple, and the dataset's run-id (UTC
timestamp). train_*.py reads from manifest.json, never directly from the
DB, so a training run can be reproduced byte-for-byte from its manifest.

Re-runnable nightly: each run produces a fresh directory; old runs are
left alone so older models can be reproduced.

Usage:
    python -m training.export_dataset --condition reflection
    python -m training.export_dataset --condition reflection --include-negatives
    python -m training.export_dataset --condition dead-fixture --out-dir /mnt/data/pretagger
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import boto3
import psycopg
from rich.console import Console

console = Console()

PRODUCTION_BUCKET = "arem-production-edit-jobs"
TRAINING_BUCKET   = "arem-training-data"


def _r2_client():
    account = os.environ["R2_ACCOUNT_ID"]
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def _fetch_bytes(client, bucket: str, key: str) -> Optional[bytes]:
    try:
        r = client.get_object(Bucket=bucket, Key=key)
        return r["Body"].read()
    except client.exceptions.NoSuchKey:
        return None
    except Exception as e:
        console.print(f"[red]r2 fetch {bucket}/{key}: {e}[/]")
        return None


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def export(condition: str, out_root: Path, include_negatives: bool) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    out_dir = out_root / condition / run_id
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    db_url = os.environ["DATABASE_URL"]
    r2 = _r2_client()

    # Query: verified positives (always) and optionally verified negatives
    where_verified = "TRUE" if include_negatives else "verified = TRUE"
    with psycopg.connect(db_url) as conn, conn.cursor() as cur:
        cur.execute(f"""
            SELECT cl.id,
                   cl."imageReviewId",
                   cl.verified,
                   cl.source,
                   cl."verifiedAt",
                   cl."verifiedBy",
                   ir."midStem",
                   ir."jobId",
                   ir."productionR2Path",
                   ir."outputR2Path",
                   j."jobFolderName",
                   j.photographer
            FROM "ConditionLabel" cl
            JOIN "ImageReview" ir ON ir.id = cl."imageReviewId"
            LEFT JOIN "Job" j ON j.id = ir."jobId"
            WHERE cl.condition = %s AND {where_verified}
              AND ir."productionR2Path" IS NOT NULL
            ORDER BY cl."verifiedAt" ASC
        """, (condition,))
        rows = cur.fetchall()

    console.print(f"[bold]{condition}[/]: {len(rows)} verified rows (negatives included: {include_negatives})")

    manifest_rows = []
    n_images_ok = 0
    n_masks_ok = 0
    n_missing = 0
    t0 = time.time()

    for (label_id, ir_id, verified, source, verified_at, verified_by,
         mid_stem, job_id, prod_path, output_path, job_folder, photographer) in rows:
        img_bytes = _fetch_bytes(r2, PRODUCTION_BUCKET, prod_path)
        if img_bytes is None:
            n_missing += 1
            continue
        img_sha = _sha(img_bytes)
        img_rel = f"images/{img_sha}.jpg"
        img_path = out_dir / img_rel
        if not img_path.exists():
            img_path.write_bytes(img_bytes)
        n_images_ok += 1

        mask_rel = None
        if condition == "reflection" and verified:
            # Reflection masks are stored separately, joined by
            # "<photographer-kebab>/<jobFolderName>/<midStem>" pairId.
            # Best-effort: query by suffix match.
            with psycopg.connect(db_url) as conn2, conn2.cursor() as cur2:
                cur2.execute("""
                    SELECT "maskR2Path" FROM "ReflectionMask"
                    WHERE "pairId" LIKE %s AND status = 'good'
                    LIMIT 1
                """, (f"%/{job_folder}/{mid_stem}",))
                mask_row = cur2.fetchone()
            if mask_row:
                mask_bytes = _fetch_bytes(r2, TRAINING_BUCKET, mask_row[0])
                if mask_bytes is not None:
                    mask_sha = _sha(mask_bytes)
                    mask_rel = f"masks/{mask_sha}.png"
                    mask_path = out_dir / mask_rel
                    if not mask_path.exists():
                        mask_path.write_bytes(mask_bytes)
                    n_masks_ok += 1

        manifest_rows.append({
            "label_id": label_id,
            "image_review_id": ir_id,
            "condition": condition,
            "verified": verified,
            "source": source,
            "verified_at": verified_at.isoformat() if verified_at else None,
            "verified_by": verified_by,
            "mid_stem": mid_stem,
            "job_id": job_id,
            "job_folder_name": job_folder,
            "photographer": photographer,
            "image_path": img_rel,
            "image_sha": img_sha,
            "mask_path": mask_rel,
            "production_r2_path": prod_path,
        })

    manifest = {
        "schema_version": 1,
        "condition": condition,
        "run_id": run_id,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "include_negatives": include_negatives,
        "totals": {
            "rows":            len(manifest_rows),
            "positives":       sum(1 for r in manifest_rows if r["verified"]),
            "negatives":       sum(1 for r in manifest_rows if not r["verified"]),
            "images_written":  n_images_ok,
            "masks_written":   n_masks_ok,
            "missing_images":  n_missing,
        },
        "rows": manifest_rows,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    elapsed = time.time() - t0
    console.print(
        f"[green]exported[/] {len(manifest_rows)} rows "
        f"({n_images_ok} images, {n_masks_ok} masks, {n_missing} missing) "
        f"in {elapsed:.1f}s → {out_dir}"
    )
    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True,
                    help="reflection | dead-fixture | photographer-shadow | finger | ...")
    ap.add_argument("--out-dir", default="./data",
                    help="root output directory (default ./data)")
    ap.add_argument("--include-negatives", action="store_true",
                    help="also export confirmed-negative rows (default: positives only)")
    args = ap.parse_args()
    export(args.condition, Path(args.out_dir), args.include_negatives)
    return 0


if __name__ == "__main__":
    sys.exit(main())
