#!/bin/bash
# Weekly retraining cycle for the pre-tagger. Invoked by
# arem-pretagger-retrain.timer.
#
# For each condition with enough verified labels:
#   1. Export fresh dataset (DB → local manifest)
#   2. Train candidate from the new manifest
#   3. Eval the candidate; compare to currently-active
#   4. If candidate beats active AND clears the bar → promote (R2 upload + local swap)
#   5. POST /reload to the running service so it picks up the new bundle
#
# If any condition's training fails or doesn't promote, the others
# still run. Existing active bundles stay in place — nothing breaks
# in production.

set -uo pipefail
cd "$(dirname "$0")/.."

if [ -f /etc/arem-pretagger.env ]; then
  set -a; source /etc/arem-pretagger.env; set +a
fi
: "${PRETAG_TOKEN:?PRETAG_TOKEN not set}"
: "${DATABASE_URL:?DATABASE_URL not set}"

PYTHON="${PYTHON_BIN:-/home/jordan/miniconda3/envs/arem-photo-ai/bin/python}"
PORT="${PORT:-8090}"

# Min verified-positives threshold before we'll even attempt retraining.
# Below this, the model would overfit; keep the previous active.
declare -A MIN_POSITIVES=(
  [reflection]=100
  [dead-fixture]=100
  [photographer-shadow]=150
  [finger]=150
)

CONDITIONS=("reflection" "dead-fixture" "photographer-shadow" "finger")
LOG_DIR="$HOME/arem-pretagger/logs/retrain-$(date -u +%Y-%m-%dT%H-%M-%SZ)"
mkdir -p "$LOG_DIR"

for cond in "${CONDITIONS[@]}"; do
  log="$LOG_DIR/$cond.log"
  echo "===== $cond =====" | tee -a "$log"

  # 1. Export
  if ! "$PYTHON" -m training.export_dataset --condition "$cond" --include-negatives 2>&1 | tee -a "$log"; then
    echo "  [export failed — skipping $cond]" | tee -a "$log"
    continue
  fi
  manifest=$(ls -td data/"$cond"/*/manifest.json 2>/dev/null | head -1)
  if [ -z "$manifest" ]; then
    echo "  [no manifest — skipping $cond]" | tee -a "$log"
    continue
  fi
  pos_count=$(jq '.totals.positives' "$manifest")
  min=${MIN_POSITIVES[$cond]:-100}
  if [ "$pos_count" -lt "$min" ]; then
    echo "  [only $pos_count positives (need $min) — skipping $cond]" | tee -a "$log"
    continue
  fi

  # 2. Train
  trainer="training.train_$(echo "$cond" | tr '-' '_')"
  if ! "$PYTHON" -m "$trainer" --manifest "$manifest" 2>&1 | tee -a "$log"; then
    echo "  [train failed — skipping $cond]" | tee -a "$log"
    continue
  fi

  bundle=$(ls -td checkpoints-staging/"$cond"/*/manifest.json 2>/dev/null | head -1 | xargs dirname)
  if [ -z "$bundle" ]; then continue; fi

  # 3. Eval
  if ! "$PYTHON" -m training.eval --bundle "$bundle" --compare-active 2>&1 | tee -a "$log"; then
    eval_exit=$?
    echo "  [eval did not promote (exit $eval_exit) — skipping $cond]" | tee -a "$log"
    continue
  fi

  # 4. Promote
  if ! "$PYTHON" scripts/promote_checkpoint.py "$cond" "$bundle" 2>&1 | tee -a "$log"; then
    echo "  [promote failed — skipping $cond]" | tee -a "$log"
    continue
  fi
done

# 5. Reload the live service so it picks up everything we promoted
curl -fsS -X POST -H "X-Pretag-Token: $PRETAG_TOKEN" "http://127.0.0.1:$PORT/reload" \
  | tee -a "$LOG_DIR/reload.log" || echo "[reload failed]"

echo "===== retraining cycle complete: $LOG_DIR ====="
