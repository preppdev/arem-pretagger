"""
arem-pretagger inference HTTP service.

POST /pretag with image bytes (multipart or base64) → per-condition
predictions (confidence + optional mask/bbox). Caller is the
arem-worker post-Stage-2 step and the dashboard's batch backfill.

Models are loaded lazily from CHECKPOINT_DIR on first request. The
model registry knows which conditions have an active checkpoint; the
response only includes those (other conditions return null so callers
can distinguish "model says no" from "no model for this condition
yet").
"""

import os
import time
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, UploadFile, File
from pydantic import BaseModel

PRETAG_TOKEN = os.environ.get("PRETAG_TOKEN", "")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "./checkpoints")

app = FastAPI(title="arem-pretagger", version="0.1.0")


class ConditionPrediction(BaseModel):
    confidence: float
    mask_r2_path: Optional[str] = None
    bboxes: Optional[list[list[float]]] = None  # [[x, y, w, h], ...] normalized 0-1


class PretagResponse(BaseModel):
    model_version: str
    latency_ms: int
    conditions: dict[str, ConditionPrediction]


def _check_auth(x_pretag_token: Optional[str]) -> None:
    if not PRETAG_TOKEN:
        raise HTTPException(status_code=503, detail="PRETAG_TOKEN not configured on server")
    if x_pretag_token != PRETAG_TOKEN:
        raise HTTPException(status_code=401, detail="invalid pretag token")


@app.get("/health")
def health() -> dict:
    """Liveness + which conditions have an active model loaded."""
    return {
        "status": "ok",
        "checkpoint_dir": CHECKPOINT_DIR,
        "active_conditions": [],  # populated once model_registry is wired up
    }


@app.post("/pretag", response_model=PretagResponse)
async def pretag(
    image: UploadFile = File(...),
    x_pretag_token: Optional[str] = Header(default=None),
) -> PretagResponse:
    _check_auth(x_pretag_token)
    t0 = time.monotonic()
    _ = await image.read()  # placeholder until model registry is wired
    return PretagResponse(
        model_version="stub-0.0.0",
        latency_ms=int((time.monotonic() - t0) * 1000),
        conditions={},
    )
