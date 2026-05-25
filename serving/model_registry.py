"""
Model registry: knows which conditions have an active checkpoint and
dispatches inference to the right backend.

A "model bundle" on disk is:
    <CHECKPOINT_DIR>/<condition>/
        manifest.json   # {"model_type": "mobile-sam" | "yolo-detect" | "classifier",
                        #  "version":    "reflection-mobilesam-v1-2026-05-28",
                        #  "weights":    "weights.pt" | "weights.pth",
                        #  "threshold":  0.5,
                        #  "trained_at": "2026-05-28T18:42:00Z",
                        #  "metrics":    {"iou": 0.81, "f1": 0.83, ...}}
        weights.pt|.pth

If a condition has no manifest.json, the registry treats it as "no
model yet" and the /pretag response simply omits that condition (NOT a
null prediction — callers can tell the difference between "model says
0.0" and "no model for this condition").

Idempotent reload: load() can be called whenever sync_checkpoints.sh
pulls a new bundle from R2. The serving process doesn't need to
restart to pick up a new model.
"""

from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class ModelBundle:
    condition: str
    model_type: str        # "mobile-sam" | "yolo-detect" | "classifier"
    version: str
    weights_path: Path
    threshold: float
    trained_at: str
    metrics: dict[str, Any]
    # Lazy: the actual model object is loaded on first inference call so
    # process startup is fast and an unused condition doesn't pay the
    # weight-load cost.
    _model: Any = None

    def ensure_loaded(self) -> Any:
        if self._model is not None:
            return self._model
        if self.model_type == "mobile-sam":
            self._model = _load_mobile_sam(self.weights_path)
        elif self.model_type == "yolo-detect":
            self._model = _load_yolo(self.weights_path)
        elif self.model_type == "classifier":
            self._model = _load_classifier(self.weights_path)
        else:
            raise ValueError(f"unknown model_type: {self.model_type}")
        return self._model


class Registry:
    def __init__(self, checkpoint_dir: str | Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.bundles: dict[str, ModelBundle] = {}
        self.loaded_at: Optional[float] = None

    def load(self) -> list[str]:
        """Scan CHECKPOINT_DIR for condition bundles. Returns list of
        condition keys with an active model. Safe to call repeatedly."""
        self.bundles = {}
        if not self.checkpoint_dir.exists():
            self.loaded_at = time.time()
            return []
        for cond_dir in sorted(self.checkpoint_dir.iterdir()):
            if not cond_dir.is_dir():
                continue
            manifest_path = cond_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            try:
                m = json.loads(manifest_path.read_text())
                weights_rel = m["weights"]
                self.bundles[cond_dir.name] = ModelBundle(
                    condition=cond_dir.name,
                    model_type=m["model_type"],
                    version=m["version"],
                    weights_path=cond_dir / weights_rel,
                    threshold=float(m.get("threshold", 0.5)),
                    trained_at=m["trained_at"],
                    metrics=m.get("metrics", {}),
                )
            except Exception as e:
                print(f"[registry] skipping {cond_dir.name}: {e}")
                continue
        self.loaded_at = time.time()
        return list(self.bundles.keys())

    def active_conditions(self) -> list[str]:
        return list(self.bundles.keys())

    def version_summary(self) -> dict[str, str]:
        return {k: b.version for k, b in self.bundles.items()}

    def predict_all(self, image_bytes: bytes) -> dict[str, dict[str, Any]]:
        """Run every loaded model against the image, return per-condition
        results. Each entry: {"confidence": float, optional
        "mask_r2_path": str, "bboxes": [[x,y,w,h], ...]}.

        For v1, the actual inference is stubbed — see the comments in
        the per-backend helpers. Once train_*.py produces real
        checkpoints, those helpers do the heavy lifting and this
        function ties them together unchanged.
        """
        out: dict[str, dict[str, Any]] = {}
        for cond, bundle in self.bundles.items():
            model = bundle.ensure_loaded()
            if bundle.model_type == "mobile-sam":
                out[cond] = _infer_mobile_sam(model, image_bytes, bundle.threshold)
            elif bundle.model_type == "yolo-detect":
                out[cond] = _infer_yolo(model, image_bytes, bundle.threshold)
            elif bundle.model_type == "classifier":
                out[cond] = _infer_classifier(model, image_bytes, bundle.threshold)
        return out


# ----- per-backend loaders + inference --------------------------------
#
# Stubs that return empty results until real training produces weights.
# Each stub is replaced one-line at a time when the corresponding
# train_*.py lands and produces a real checkpoint. The registry +
# /pretag plumbing don't change.


def _load_mobile_sam(weights_path: Path) -> Any:
    # TODO: from mobile_sam import sam_model_registry; load checkpoint
    return {"weights": str(weights_path), "stub": True}


def _load_yolo(weights_path: Path) -> Any:
    # TODO: from ultralytics import YOLO; YOLO(str(weights_path))
    return {"weights": str(weights_path), "stub": True}


def _load_classifier(weights_path: Path) -> Any:
    # TODO: torch.load(weights_path, map_location="cuda")
    return {"weights": str(weights_path), "stub": True}


def _infer_mobile_sam(model: Any, image_bytes: bytes, threshold: float) -> dict[str, Any]:
    # TODO real inference: decode image → run MobileSAM → mask + score.
    # When a mask is produced, the serving layer uploads it to R2
    # (pretag-masks/<id>.png) and includes mask_r2_path in the response.
    return {"confidence": 0.0, "mask_r2_path": None, "stub": True}


def _infer_yolo(model: Any, image_bytes: bytes, threshold: float) -> dict[str, Any]:
    # TODO: YOLO detect → per-fixture bboxes + on/off classification head.
    return {"confidence": 0.0, "bboxes": [], "stub": True}


def _infer_classifier(model: Any, image_bytes: bytes, threshold: float) -> dict[str, Any]:
    # TODO: small CNN binary classifier; sigmoid → confidence.
    return {"confidence": 0.0, "stub": True}
