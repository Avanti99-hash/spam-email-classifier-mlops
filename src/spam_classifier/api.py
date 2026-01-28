from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from .predict import load_artifacts, predict_batch, predict_text
from .service_state import Stats

load_dotenv()

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"

# Simple admin auth via API key header
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")  # set this in .env or environment

app = FastAPI(title="Spam Email Classifier API")

stats = Stats()

# Keep model in memory for performance (optional)
_cached = {"w2v": None, "model": None, "model_name": None}


def require_admin(x_api_key: Optional[str] = Header(default=None)) -> None:
    # If you didn't set ADMIN_API_KEY, we block admin endpoints (safer default)
    if not ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Admin key not configured")

    if not x_api_key or x_api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


class PredictRequest(BaseModel):
    text: str


class PredictBatchRequest(BaseModel):
    texts: List[str]


@app.on_event("startup")
def startup_load_model():
    # Try loading artifacts on startup; if missing, API still runs but predict will fail clearly
    try:
        w2v, model_name, model = load_artifacts(ARTIFACTS_DIR)
        _cached["w2v"] = w2v
        _cached["model_name"] = model_name
        _cached["model"] = model
    except Exception:
        # Don't crash service; allows health/metrics to work even before training
        pass


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def get_stats():
    return stats.snapshot()


@app.get("/metrics")
def get_metrics():
    metrics_file = ARTIFACTS_DIR / "metrics.json"
    if not metrics_file.exists():
        raise HTTPException(status_code=404, detail="metrics.json not found. Train the model first.")
    return json.loads(metrics_file.read_text(encoding="utf-8"))


@app.get("/model/info")
def model_info():
    model_file = ARTIFACTS_DIR / "model.joblib"
    w2v_file = ARTIFACTS_DIR / "word2vec.model"

    if not model_file.exists() or not w2v_file.exists():
        raise HTTPException(status_code=404, detail="Model artifacts not found. Train the model first.")

    info = {
        "artifacts_dir": str(ARTIFACTS_DIR),
        "model_loaded_in_memory": _cached["model"] is not None,
        "model_name": _cached["model_name"],
        "has_word2vec": w2v_file.exists(),
    }

    # If metrics exist, include best metrics summary
    metrics_file = ARTIFACTS_DIR / "metrics.json"
    if metrics_file.exists():
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        info["best_model"] = metrics.get("best_model")
        info["best_metrics"] = metrics.get("best_metrics")

    return info


@app.post("/predict")
def predict(req: PredictRequest):
    start = time.time()
    try:
        out = predict_text(req.text, ARTIFACTS_DIR)
        latency_ms = (time.time() - start) * 1000.0
        stats.record_request("predict", latency_ms, predicted_label=out.get("label"))
        return out
    except Exception as e:
        stats.record_error()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch_endpoint(req: PredictBatchRequest):
    start = time.time()
    try:
        if len(req.texts) == 0:
            raise HTTPException(status_code=400, detail="texts list is empty")
        if len(req.texts) > 256:
            raise HTTPException(status_code=400, detail="Max batch size is 256")

        out = predict_batch(req.texts, ARTIFACTS_DIR)
        latency_ms = (time.time() - start) * 1000.0

        # Count spam/ham from batch
        for item in out:
            stats.record_request("predict_batch", 0.0, predicted_label=item.get("label"))
        # Record one latency measurement for the whole batch request
        stats.record_request("predict_batch", latency_ms, predicted_label=None)

        return out
    except HTTPException:
        stats.record_error()
        raise
    except Exception as e:
        stats.record_error()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/reload")
def admin_reload(_: None = Depends(require_admin)):
    """
    Admin-only: reload artifacts from disk without restarting the service.
    Caller must send header: X-API-Key: <ADMIN_API_KEY>
    """
    model_file = ARTIFACTS_DIR / "model.joblib"
    w2v_file = ARTIFACTS_DIR / "word2vec.model"
    if not model_file.exists() or not w2v_file.exists():
        raise HTTPException(status_code=404, detail="Model artifacts not found. Train the model first.")

    try:
        w2v, model_name, model = load_artifacts(ARTIFACTS_DIR)
        _cached["w2v"] = w2v
        _cached["model_name"] = model_name
        _cached["model"] = model
        stats.mark_reload()
        return {"status": "reloaded", "model_name": model_name}
    except Exception as e:
        stats.record_error()
        raise HTTPException(status_code=500, detail=str(e))
