"""
arem-pretagger inference HTTP service.

POST /pretag with image bytes → per-condition predictions (confidence
+ optional mask R2 path + optional bboxes). Caller is the arem-worker
post-Stage-2 step.

Models live under CHECKPOINT_DIR (default ./checkpoints/<condition>/).
The registry hot-reloads on /reload — pulling a fresh bundle from R2
and pointing the in-process registry at it doesn't need a restart.
"""

import os
import time
from typing import Optional

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel

from serving.model_registry import Registry

PRETAG_TOKEN = os.environ.get("PRETAG_TOKEN", "")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "./checkpoints")

app = FastAPI(title="arem-pretagger", version="0.2.0")
registry = Registry(CHECKPOINT_DIR)
registry.load()


class ConditionPrediction(BaseModel):
    confidence: float
    mask_r2_path: Optional[str] = None
    bboxes: Optional[list[list[float]]] = None
    stub: Optional[bool] = None  # true while the bundle is the placeholder


class PretagResponse(BaseModel):
    model_versions: dict[str, str]
    latency_ms: int
    conditions: dict[str, ConditionPrediction]


def _check_auth(token: Optional[str]) -> None:
    if not PRETAG_TOKEN:
        raise HTTPException(status_code=503, detail="PRETAG_TOKEN not configured on server")
    if token != PRETAG_TOKEN:
        raise HTTPException(status_code=401, detail="invalid pretag token")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "checkpoint_dir": str(registry.checkpoint_dir),
        "active_conditions": registry.active_conditions(),
        "model_versions": registry.version_summary(),
        "loaded_at": registry.loaded_at,
    }


@app.post("/reload")
def reload_models(x_pretag_token: Optional[str] = Header(default=None)) -> dict:
    _check_auth(x_pretag_token)
    active = registry.load()
    return {"reloaded": True, "active_conditions": active, "model_versions": registry.version_summary()}


@app.post("/pretag", response_model=PretagResponse)
async def pretag(
    image: UploadFile = File(...),
    x_pretag_token: Optional[str] = Header(default=None),
) -> PretagResponse:
    _check_auth(x_pretag_token)
    t0 = time.monotonic()
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="empty image")
    raw = registry.predict_all(image_bytes)
    return PretagResponse(
        model_versions=registry.version_summary(),
        latency_ms=int((time.monotonic() - t0) * 1000),
        conditions={
            cond: ConditionPrediction(
                confidence=v.get("confidence", 0.0),
                mask_r2_path=v.get("mask_r2_path"),
                bboxes=v.get("bboxes"),
                stub=v.get("stub"),
            )
            for cond, v in raw.items()
        },
    )
