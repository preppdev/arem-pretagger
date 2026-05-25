# arem-pretagger

ML inference service that pre-tags real-estate photos with per-condition
predictions (reflection masks, dead-fixture bboxes, etc.) before they
reach the human reviewer in the `arem-editing` dashboard.

Runs on the same Ubuntu 3090 box as `arem-worker`, as a separate
systemd-managed service. Models live in R2; deploys are a model-version
bump + `update.sh`.

## What it does

For each delivered image, the service returns:

```jsonc
{
  "modelVersion": "reflection-v1-2026-05-25",
  "latencyMs": 42,
  "conditions": {
    "reflection":    { "confidence": 0.91, "maskR2Path": "pretag/<job>/<mid>_reflection.png" },
    "deadFixture":   { "confidence": 0.04, "bboxes": [] },
    "photographerShadow": { "confidence": 0.12 },
    "finger":        { "confidence": 0.02 }
  }
}
```

The dashboard worker (`arem-worker` post-Stage-2 step) POSTs the
deliverable to `POST /pretag`, writes the response onto
`ImageReview.preTagConfidence` / `preTagModelVersion`, and the reviewer
UI renders the pre-tags as suggestions. Reviewer confirmations and
overrides are captured as `PreTagFeedback` rows in the dashboard DB and
flow back into the next training cycle.

## Layout

```
arem-pretagger/
  serving/             # FastAPI inference service
    app.py
    model_registry.py  # loads checkpoints from local cache; falls back to R2
  training/            # training scripts (run on the 3090, ad-hoc + cron)
    export_dataset.py  # DB → local manifest (images + masks + labels)
    train_reflection.py
    train_dead_fixture.py
    eval.py            # evaluate a candidate checkpoint vs. the prior version
  scripts/
    sync_checkpoints.sh    # pull latest active model versions from R2
    promote_checkpoint.py  # mark a candidate as the active version in R2
  systemd/
    arem-pretagger.service
  provision.sh         # one-shot install on a fresh box
  update.sh            # pull + reinstall + restart
  run_serve.sh         # systemd ExecStart — loads env, execs uvicorn
  requirements.txt     # deltas over the arem-photo-ai conda env
```

## Conda env

Shares the existing `~/miniconda3/envs/arem-photo-ai` environment with
`arem-worker` rather than building a parallel one. Avoids disk
duplication and CUDA-driver mismatch risk. Adds:

- `fastapi` + `uvicorn[standard]`
- `ultralytics` (YOLO for dead-fixture detector)
- `segment_anything` + MobileSAM fork (reflection segmenter)

These get installed via `provision.sh` (idempotent `pip install`).

## Service ports + auth

Listens on `0.0.0.0:8090` on the LAN at `10.2.0.15`. No public exposure.
Caller auth via `X-Pretag-Token` header matched against
`PRETAG_TOKEN` in `/etc/arem-pretagger.env`.

## Why a separate repo

Hard isolation between "thing that produces deliverables" (arem-worker)
and "thing that classifies them" (arem-pretagger). A bad pre-tagger
deploy can't break the production editing pipeline. Independent
retrain cadence + rollback path.
