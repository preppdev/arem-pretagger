# Publish guide — bringing the pre-tagger online

Everything else is built and committed. The remaining manual work is
in this file. Five steps; ~30 minutes of your time (most of which is
the verification pass).

## 1. Verify some labels (browser, ~1 hour)

Open https://arem-editing-dashboard.vercel.app/verify-labels and work
through:

| Condition           | Items to verify | Notes |
|---------------------|-----------------|-------|
| reflection          |  21 stragglers  | The 2026-05-22 curation pass left 21 unverified since |
| dead-fixture        | 188             | Full pass needed |
| photographer-shadow |  39             | Will be a small training set — collect more via "negatives" mode |
| finger              |   0             | No reviewer-flagged data — collect via "negatives" mode (look through random images, flag any with fingers) |

Use the "Sample missed positives" mode to find false-negatives (the
queue is random, so just keep clicking until you have ~100 confirmed
clean negatives per condition for balanced training).

## 2. Install arem-pretagger on the 3090 box (one command, ~10 min)

```bash
ssh jordan@10.2.0.15

# Clone
git clone https://github.com/preppdev/arem-pretagger ~/arem-pretagger

# Generate the env file. PRETAG_TOKEN is a fresh peer-secret between
# arem-worker and arem-pretagger; both need it.
PRETAG_TOKEN=$(openssl rand -hex 32)
sudo tee /etc/arem-pretagger.env > /dev/null <<EOF
PRETAG_TOKEN=$PRETAG_TOKEN
$(sudo grep -E '^(DATABASE_URL|R2_ACCOUNT_ID|R2_ACCESS_KEY_ID|R2_SECRET_ACCESS_KEY)=' /etc/arem-worker.env)
EOF
sudo chown jordan:jordan /etc/arem-pretagger.env
sudo chmod 0640 /etc/arem-pretagger.env

# Also append PRETAG_TOKEN to the worker's env so it can authenticate
# when calling localhost:8090/pretag.
echo "PRETAG_TOKEN=$PRETAG_TOKEN" | sudo tee -a /etc/arem-worker.env > /dev/null

# Install
bash ~/arem-pretagger/provision.sh

# Restart the worker so it picks up PRETAG_TOKEN
sudo systemctl restart arem-worker-local
```

After this:
- `systemctl status arem-pretagger.service` → active (running)
- `curl http://127.0.0.1:8090/health` → `active_conditions: []` (no models yet)
- Worker journal will start logging `pretag: posted 38/38 (skipped 0)`
  on each new shoot — but every condition response is `{confidence:
  0, stub: true}` until a real model is trained.

## 3. Train the first model — reflection (~30 min on the 3090)

Once you've verified ≥100 reflection positives in step 1 (you already
have 241 from the legacy curation + however many you verified today):

```bash
# Pull verified labels into a local training manifest
~/miniconda3/envs/arem-photo-ai/bin/python -m training.export_dataset \
  --condition reflection --include-negatives

# Replace _stub_train() in training/train_reflection.py with the
# documented MobileSAM fine-tune loop (the file has the reference code
# inline as comments). Commit + push.

# Run training
~/miniconda3/envs/arem-photo-ai/bin/python -m training.train_reflection \
  --manifest ./data/reflection/<run-id>/manifest.json

# Eval against the promotion bar (IoU ≥ 0.65)
~/miniconda3/envs/arem-photo-ai/bin/python -m training.eval \
  --bundle ./checkpoints-staging/reflection/<run-id>

# Promote → uploads to R2 + swaps local active + emits a ready-to-paste
# curl command for the next step.
~/miniconda3/envs/arem-photo-ai/bin/python scripts/promote_checkpoint.py \
  reflection ./checkpoints-staging/reflection/<run-id>

# Hot-reload the live service
curl -X POST -H "X-Pretag-Token: $(grep PRETAG_TOKEN /etc/arem-pretagger.env | cut -d= -f2)" \
     http://127.0.0.1:8090/reload
```

## 4. (Optional) Same for dead-fixture, etc.

Same flow with `--condition dead-fixture` and
`training/train_dead_fixture.py`. Replace the stub training loop with
ultralytics YOLOv8n training (one-liner per the file's comments).

## 5. Watch the loop run itself

The weekly retraining cron is already installed:

```bash
systemctl list-timers arem-pretagger-retrain
```

Sunday 03:00 local, each condition with ≥100 new verified labels gets
exported → trained → eval'd → auto-promoted (only if it beats the
active model on the held-out set). The `/reload` POST fires
automatically.

To trigger a retrain cycle out of band:

```bash
sudo systemctl start arem-pretagger-retrain.service
journalctl -u arem-pretagger-retrain -f
```

## How to undo if anything goes sideways

```bash
# Disable the service (worker keeps writing null preTagConfidence)
sudo systemctl disable --now arem-pretagger.service
sudo systemctl disable --now arem-pretagger-retrain.timer

# OR: remove PRETAG_TOKEN from /etc/arem-worker.env so the worker stops
# calling the pretagger. Reviewer UI just stops showing suggestions.

# Rollback to the previous model bundle for one condition
~/miniconda3/envs/arem-photo-ai/bin/python -c "
import boto3, os
from pathlib import Path
client = boto3.client('s3', endpoint_url=f'https://{os.environ[\"R2_ACCOUNT_ID\"]}.r2.cloudflarestorage.com',
                     aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],
                     aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'], region_name='auto')
for obj in client.list_objects_v2(Bucket='arem-training-data',
                                  Prefix='pretagger-models/reflection/previous/').get('Contents', []):
    new_key = obj['Key'].replace('/previous/', '/active/')
    client.copy_object(Bucket='arem-training-data',
                      CopySource={'Bucket': 'arem-training-data', 'Key': obj['Key']},
                      Key=new_key)
"
bash ~/arem-pretagger/scripts/sync_checkpoints.sh
curl -X POST -H "X-Pretag-Token: $PRETAG_TOKEN" http://127.0.0.1:8090/reload
```

## What's intentionally NOT done yet

- **Real training loops.** Both `train_reflection.py` and
  `train_dead_fixture.py` ship with `_stub_train()` placeholders. The
  reference code is inline in each file as comments — substituting it
  is the only meaningful coding step left. Until then, all conditions
  return `confidence: 0` and the reviewer UI shows no suggestions.
- **Finger condition.** Zero reviewer-flagged data. Bootstrap by
  going through ~500 random images in negatives-mode of
  /verify-labels and flagging any you see, then train as usual.
- **Sub-condition tuning.** PRETAG_SUGGESTION_THRESHOLD is currently
  one global value (0.5). Once we have real per-condition calibration
  curves, swap it for a per-condition map in lib/settings-defaults.ts.
