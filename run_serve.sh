#!/bin/bash
# Launch the arem-pretagger FastAPI inference service on the 3090.
#
# Use cases:
#   bash run_serve.sh        # foreground (Ctrl-C to stop)
#   systemd ExecStart        # arem-pretagger.service invokes this

set -euo pipefail
cd "$(dirname "$0")"

# Load credentials. When invoked via systemd, EnvironmentFile already
# injects these. For manual invocations, source the same file.
if [ -f /etc/arem-pretagger.env ]; then
  set -a; source /etc/arem-pretagger.env; set +a
fi

: "${PRETAG_TOKEN:?PRETAG_TOKEN not set — see systemd/arem-pretagger.service for env file install}"

export PORT="${PORT:-8090}"
export HOST="${HOST:-0.0.0.0}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-$(pwd)/checkpoints}"
export PYTHON_BIN="${PYTHON_BIN:-/home/jordan/miniconda3/envs/arem-photo-ai/bin/python}"

echo "[arem-pretagger] host=$HOST port=$PORT"
echo "[arem-pretagger] checkpoint_dir=$CHECKPOINT_DIR"
echo "[arem-pretagger] python=$PYTHON_BIN"
echo

exec "$PYTHON_BIN" -m uvicorn serving.app:app \
  --host "$HOST" \
  --port "$PORT" \
  --workers 1 \
  --log-level info
