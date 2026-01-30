from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from gensim.models import Word2Vec

logger = logging.getLogger(__name__)


def load_artifacts(artifacts_dir: Path) -> Tuple[Word2Vec, str, object]:
    """
    Loads Word2Vec + classifier from artifacts directory.
    Logs success/failure with enough detail to debug CI/CD deployments.
    """
    t0 = time.time()
    w2v_path = artifacts_dir / "word2vec.model"
    model_path = artifacts_dir / "model.joblib"

    logger.info("Loading artifacts | dir=%s", artifacts_dir)

    if not w2v_path.exists():
        logger.error("Missing Word2Vec artifact: %s", w2v_path)
        raise FileNotFoundError(f"Missing Word2Vec artifact: {w2v_path}")

    if not model_path.exists():
        logger.error("Missing model artifact: %s", model_path)
        raise FileNotFoundError(f"Missing model artifact: {model_path}")

    try:
        logger.debug("Loading Word2Vec from %s", w2v_path)
        w2v = Word2Vec.load(str(w2v_path))

        logger.debug("Loading classifier from %s", model_path)
        model_obj = joblib.load(model_path)

        model = model_obj["model"]
        model_name = model_obj.get("model_name", model.__class__.__name__)

        logger.info(
            "Artifacts loaded | model_name=%s model_type=%s vector_size=%d | in %.2fms",
            model_name,
            model.__class__.__name__,
            w2v.vector_size,
            (time.time() - t0) * 1000.0,
        )
        return w2v, model_name, model

    except Exception:
        logger.exception("Failed to load artifacts from %s", artifacts_dir)
        raise


def vectorize_single_text(text: str, w2v: Word2Vec) -> np.ndarray:
    """
    Converts a single string into one Word2Vec-average vector.
    Logs only in DEBUG to avoid log spam.
    """
    if text is None:
        logger.debug("vectorize_single_text received None; using empty string")
        text = ""

    tokens = str(text).split()
    token_vectors = [w2v.wv[w] for w in tokens if w in w2v.wv]

    if not token_vectors:
        logger.debug("No in-vocab tokens found; returning zero vector | tokens=%d", len(tokens))
        v = np.zeros(w2v.vector_size)
    else:
        v = np.mean(token_vectors, axis=0)

    return v.reshape(1, -1)


def predict_text(text: str, artifacts_dir: Path) -> Dict:
    """
    Predicts spam/ham for a single text.
    Logs inference latency and whether probability is available.
    """
    t0 = time.time()
    try:
        w2v, model_name, model = load_artifacts(artifacts_dir)
        X = vectorize_single_text(text, w2v)

        pred = int(model.predict(X)[0])

        score = None
        has_proba = hasattr(model, "predict_proba")
        if has_proba:
            score = float(model.predict_proba(X)[0][pred])

        out = {"model": model_name, "label": "Spam" if pred == 1 else "Ham", "score": score}

        logger.info(
            "Predict completed | label=%s has_proba=%s | in %.2fms",
            out["label"],
            has_proba,
            (time.time() - t0) * 1000.0,
        )
        return out

    except Exception:
        logger.exception("Predict failed")
        raise


def vectorize_texts_batch(texts: List[str], w2v: Word2Vec) -> np.ndarray:
    """
    Batch vectorization helper.
    """
    vecs = []
    for t in texts:
        vecs.append(vectorize_single_text(t, w2v)[0])
    X = np.array(vecs)
    logger.debug("Batch vectorization completed | shape=%s", X.shape)
    return X


def predict_batch(texts: List[str], artifacts_dir: Path) -> List[Dict]:
    """
    Predicts for a list of texts.
    Logs batch size + latency, and counts predicted spam/ham.
    """
    t0 = time.time()

    if texts is None:
        logger.error("predict_batch received None texts")
        raise ValueError("texts cannot be None")

    logger.info("Batch predict started | batch_size=%d", len(texts))

    try:
        w2v, model_name, model = load_artifacts(artifacts_dir)
        X = vectorize_texts_batch(texts, w2v)

        preds = model.predict(X).astype(int).tolist()

        scores = None
        has_proba = hasattr(model, "predict_proba")
        if has_proba:
            prob = model.predict_proba(X)
            scores = [float(prob[i][preds[i]]) for i in range(len(preds))]

        out = []
        spam_count = 0
        ham_count = 0

        for i, p in enumerate(preds):
            label = "Spam" if p == 1 else "Ham"
            if label == "Spam":
                spam_count += 1
            else:
                ham_count += 1

            out.append(
                {
                    "model": model_name,
                    "label": label,
                    "score": None if scores is None else scores[i],
                }
            )

        logger.info(
            "Batch predict completed | spam=%d ham=%d has_proba=%s | in %.2fms",
            spam_count,
            ham_count,
            has_proba,
            (time.time() - t0) * 1000.0,
        )
        return out

    except Exception:
        logger.exception("Batch predict failed")
        raise
