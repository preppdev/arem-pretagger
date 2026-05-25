#!/bin/bash
# Idempotent provision for arem-pretagger on the 3090 box.
# Run after the arem-worker box is already provisioned — we reuse its
# conda env + R2/Dropbox credentials.
#
#   curl -sSL https://raw.githubusercontent.com/preppdev/arem-pretagger/main/provision.sh | bash
#
# Or, from a local clone:
#   bash ~/arem-pretagger/provision.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/arem-pretagger}"
CONDA_ENV="${CONDA_ENV:-arem-photo-ai}"
CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"

echo "[provision] checking prerequisites…"
command -v "$CONDA_BIN" >/dev/null || { echo "miniconda not found at $CONDA_BIN — run arem-worker provision.sh first"; exit 1; }
[ -d "$REPO_DIR" ] || { echo "repo not found at $REPO_DIR — clone it first"; exit 1; }

echo "[provision] installing pip deps into $CONDA_ENV…"
"$HOME/miniconda3/envs/$CONDA_ENV/bin/pip" install --upgrade -r "$REPO_DIR/requirements.txt"

echo "[provision] checking credentials gate…"
if [ ! -f /etc/arem-pretagger.env ]; then
  cat >&2 <<EOF

  /etc/arem-pretagger.env is missing. Stage it and re-run.

  Minimum required:
    PRETAG_TOKEN=<random; matches the same name in arem-editing Vercel env>
    DATABASE_URL=<same as arem-worker>
    R2_ACCOUNT_ID=...
    R2_ACCESS_KEY_ID=...
    R2_SECRET_ACCESS_KEY=...

  Install with:
    sudo install -m 0640 -o jordan -g jordan /tmp/arem-pretagger.env /etc/arem-pretagger.env

EOF
  exit 2
fi

echo "[provision] installing systemd unit…"
sudo cp "$REPO_DIR/systemd/arem-pretagger.service" /etc/systemd/system/arem-pretagger.service
sudo systemctl daemon-reload
sudo systemctl enable --now arem-pretagger.service

echo "[provision] done. Checking service status:"
sleep 2
systemctl --no-pager status arem-pretagger.service | head -15

echo
echo "Health check:"
curl -sS "http://127.0.0.1:${PORT:-8090}/health" | head
echo
