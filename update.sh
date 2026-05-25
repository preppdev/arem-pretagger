#!/bin/bash
# Pull the latest arem-pretagger commits + re-sync deps + restart.
# Run on the 3090 box.

set -euo pipefail
cd "$(dirname "$0")"

echo "[update] git pull…"
git pull --ff-only

echo "[update] re-installing pip deps…"
"$HOME/miniconda3/envs/arem-photo-ai/bin/pip" install -r requirements.txt

# Re-install systemd unit if it changed
if ! sudo cmp -s systemd/arem-pretagger.service /etc/systemd/system/arem-pretagger.service; then
  echo "[update] systemd unit changed — reinstalling"
  sudo cp systemd/arem-pretagger.service /etc/systemd/system/arem-pretagger.service
  sudo systemctl daemon-reload
fi

echo "[update] restart service"
sudo systemctl restart arem-pretagger.service

sleep 2
systemctl --no-pager status arem-pretagger.service | head -10
