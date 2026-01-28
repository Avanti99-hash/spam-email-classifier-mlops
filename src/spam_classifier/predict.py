from __future__ import annotations

from pathlib import Path
from typing import Dict
from typing import List, Tuple

import joblib
import numpy as np
from gensim.models import Word2Vec


def load_artifacts(artifacts_dir: Path):
    w2v = Word2Vec.load(str(artifacts_dir / "word2vec.model"))
    model_obj = joblib.load(artifacts_dir / "model.joblib")
    model = model_obj["model"]
    model_name = model_obj["model_name"]
    return w2v, model_name, model


def vectorize_single_text(text: str, w2v: Word2Vec) -> np.ndarray:
    tokens = str(text).split()
    token_vectors = [w2v.wv[w] for w in tokens if w in w2v.wv]
    if token_vectors:
        v = np.mean(token_vectors, axis=0)
    else:
        v = np.zeros(w2v.vector_size)
    return v.reshape(1, -1)


def predict_text(text: str, artifacts_dir: Path) -> Dict:
    w2v, model_name, model = load_artifacts(artifacts_dir)
    X = vectorize_single_text(text, w2v)
    pred = int(model.predict(X)[0])

    score = None
    if hasattr(model, "predict_proba"):
        score = float(model.predict_proba(X)[0][pred])

    return {"model": model_name, "label": "Spam" if pred == 1 else "Ham", "score": score}

def vectorize_texts_batch(texts: List[str], w2v: Word2Vec) -> np.ndarray:
    vecs = []
    for t in texts:
        vecs.append(vectorize_single_text(t, w2v)[0])
    return np.array(vecs)


def predict_batch(texts: List[str], artifacts_dir: Path) -> List[Dict]:
    w2v, model_name, model = load_artifacts(artifacts_dir)
    X = vectorize_texts_batch(texts, w2v)

    preds = model.predict(X).astype(int).tolist()

    scores = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)
        scores = [float(prob[i][preds[i]]) for i in range(len(preds))]

    out = []
    for i, p in enumerate(preds):
        out.append(
            {
                "model": model_name,
                "label": "Spam" if p == 1 else "Ham",
                "score": None if scores is None else scores[i],
            }
        )
    return out
