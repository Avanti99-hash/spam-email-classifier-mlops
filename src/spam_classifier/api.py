from __future__ import annotations

import json
import os
import time
import logging
from contextlib import asynccontextmanager
from .logging_config import setup_logging
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from .predict import load_artifacts, predict_batch, predict_text
from .service_state import Stats

setup_logging(app_name="spam_classifier_api")
logger = logging.getLogger(__name__)

load_dotenv()

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"

# Simple admin auth via API key header
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")  # set this in .env or environment

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API lifespan startup initiated")

    try:
        w2v, model_name, model = load_artifacts(ARTIFACTS_DIR)
        _cached["w2v"] = w2v
        _cached["model_name"] = model_name
        _cached["model"] = model

        logger.info("Model preloaded into memory | model_name=%s vector_size=%d",model_name,w2v.vector_size,)

    except Exception as e:
        logger.warning("Model preload skipped | reason=%s | API will still start",str(e),)

    yield  # ---- application runs here ----

    logger.info("API lifespan shutdown initiated")

app = FastAPI(title="Spam Email Classifier API",lifespan=lifespan,)

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

@app.get("/health")
def health():
    logger.debug("Health check requested")
    return {"status": "ok"}


@app.get("/stats")
def get_stats():
    logger.info("Stats requested")
    snapshot = stats.snapshot()
    logger.debug("Stats snapshot: %s", snapshot)
    return snapshot


@app.get("/metrics")
def get_metrics():
    logger.info("Metrics endpoint called")
    metrics_file = ARTIFACTS_DIR / "metrics.json"
    if not metrics_file.exists():
        logger.warning("metrics.json not found")
        raise HTTPException(status_code=404, detail="metrics.json not found. Train the model first.")

    logger.info("metrics.json loaded successfully")    
    return json.loads(metrics_file.read_text(encoding="utf-8"))


@app.get("/model/info")
def model_info():
    logger.info("Model info requested")
    model_file = ARTIFACTS_DIR / "model.joblib"
    w2v_file = ARTIFACTS_DIR / "word2vec.model"

    if not model_file.exists() or not w2v_file.exists():
        logger.warning("Model info requested but artifacts missing")
        raise HTTPException(status_code=404, detail="Model artifacts not found. Train the model first.")

    info = {
        "artifacts_dir": str(ARTIFACTS_DIR),
        "model_loaded_in_memory": _cached["model"] is not None,
        "model_name": _cached["model_name"],
        "has_word2vec": w2v_file.exists(),
    }

    logger.info("Model info returned | model_loaded=%s model_name=%s",info["model_loaded_in_memory"],info["model_name"],)

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
    logger.info("Predict request received | text_length=%d", len(req.text))
    try:
        out = predict_text(req.text, ARTIFACTS_DIR)
        latency_ms = (time.time() - start) * 1000.0
        stats.record_request("predict", latency_ms, predicted_label=out.get("label"))
        logger.info("Predict success | label=%s latency_ms=%.2f",out["label"],latency_ms,)
        return out
    except Exception as e:
        stats.record_error()
        logger.exception("Predict failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch_endpoint(req: PredictBatchRequest):
    start = time.time()
    batch_size = len(req.texts)
    logger.info("Batch predict request | batch_size=%d", batch_size)
    try:
        if len(req.texts) == 0:
            logger.warning("Batch predict rejected | empty batch")
            raise HTTPException(status_code=400, detail="texts list is empty")
        if len(req.texts) > 256:
            logger.warning("Batch predict rejected | batch too large (%d)", batch_size)
            raise HTTPException(status_code=400, detail="Max batch size is 256")

        out = predict_batch(req.texts, ARTIFACTS_DIR)
        latency_ms = (time.time() - start) * 1000.0

        # Count spam/ham from batch
        for item in out:
            stats.record_request("predict_batch", 0.0, predicted_label=item.get("label"))
        # Record one latency measurement for the whole batch request
        stats.record_request("predict_batch", latency_ms, predicted_label=None)
        logger.info("Batch predict success | batch_size=%d latency_ms=%.2f",batch_size,latency_ms,)

        return out
    except HTTPException:
        stats.record_error()
        raise
    except Exception as e:
        stats.record_error()
        logger.exception("Batch predict failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/reload")
def admin_reload(_: None = Depends(require_admin)):
    logger.warning("Admin reload requested")
    """
    Admin-only: reload artifacts from disk without restarting the service.
    Caller must send header: X-API-Key: <ADMIN_API_KEY>
    """
    model_file = ARTIFACTS_DIR / "model.joblib"
    w2v_file = ARTIFACTS_DIR / "word2vec.model"
    if not model_file.exists() or not w2v_file.exists():
        logger.error("Admin reload failed | artifacts missing")
        raise HTTPException(status_code=404, detail="Model artifacts not found. Train the model first.")

    try:
        w2v, model_name, model = load_artifacts(ARTIFACTS_DIR)
        _cached["w2v"] = w2v
        _cached["model_name"] = model_name
        _cached["model"] = model
        stats.mark_reload()
        logger.warning("Admin reload completed | model_name=%s vector_size=%d",model_name,w2v.vector_size,)
        return {"status": "reloaded", "model_name": model_name}
    except Exception as e:
        stats.record_error()
        logger.exception("Admin reload failed")
        raise HTTPException(status_code=500, detail=str(e))
