#!/bin/bash
# Pull the currently-active model bundles from R2 into ./checkpoints/.
# Run after a fresh deploy, or whenever the pretagger service starts.
# Idempotent — rclone copy diffs by size+modtime.
#
# Each active bundle lives at:
#   arem-training-data:pretagger-models/<condition>/active/
#     - manifest.json
#     - weights.pt (or .pth)
#
# promote_checkpoint.py is what writes to that path; this script just
# pulls the result back down.

set -euo pipefail
cd "$(dirname "$0")/.."

CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints}"
REMOTE="${PRETAG_REMOTE:-arem-training-data:pretagger-models}"

mkdir -p "$CHECKPOINT_DIR"

CONDITIONS=("reflection" "dead-fixture" "photographer-shadow" "finger")
for cond in "${CONDITIONS[@]}"; do
  echo "[sync] $cond"
  rclone copy --no-traverse "$REMOTE/$cond/active/" "$CHECKPOINT_DIR/$cond/" \
    --include "manifest.json" \
    --include "weights.*" \
    || echo "  (no active bundle for $cond yet — skipping)"
done

echo
echo "[sync] done. Active bundles:"
find "$CHECKPOINT_DIR" -name manifest.json -maxdepth 2 -mindepth 2 -print
