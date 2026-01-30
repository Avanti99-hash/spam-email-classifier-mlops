from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


SPAM_KEYWORDS = [
    "win", "free", "urgent", "offer", "loan", "click", "subscribe",
    "money", "cash", "credit", "gift", "winner", "deal", "buy now",
    "limited time", "discount", "promotion", "guarantee", "trial", "stop", "end",
    "exclusive", "claim now",
]


def classify_email(subject: str, body: str) -> int:
    if not subject and not body:
        logger.debug("classify_email received empty subject and body")
    content = f"{subject} {body}".lower()
    return int(any(k in content for k in SPAM_KEYWORDS))


def train_word2vec(tokenized_texts: pd.Series, vector_size: int = 100) -> Word2Vec:
    logger.info("Training Word2Vec | docs=%d vector_size=%d",len(tokenized_texts),vector_size,)

    model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=vector_size,
        window=5,
        min_count=1,
        workers=4,
    )

    logger.info("Word2Vec training completed | vocab_size=%d", len(model.wv))
    return model


def vectorize_texts(texts: pd.Series, model: Word2Vec) -> np.ndarray:
    logger.info("Vectorizing texts | count=%d", len(texts))
    vectors = []
    for text in texts.astype(str):
        tokens = text.split()
        token_vectors = [model.wv[w] for w in tokens if w in model.wv]
        if token_vectors:
            vectors.append(np.mean(token_vectors, axis=0))
        else:
            vectors.append(np.zeros(model.vector_size))

    X = np.array(vectors)
    logger.info("Vectorization completed | shape=%s", X.shape)
    return X


def train_models(X_train: np.ndarray, y_train: pd.Series):
    models = {
        "RandomForest": RandomForestClassifier(random_state=42, n_estimators=100),
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
        "SVM": SVC(random_state=42, probability=True),
    }
    logger.info("Training models | models=%s", list(models.keys()))
    for m in models.values():
        m.fit(X_train, y_train)
        logger.info("Model trained successfully | model=%s", m.__class__.__name__)
    return models


def save_artifacts(artifacts_dir: Path, w2v: Word2Vec, model_name: str, model) -> None:
    logger.info("Saving artifacts | artifacts_dir=%s model=%s",artifacts_dir,model_name,)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    w2v.save(str(artifacts_dir / "word2vec.model"))
    logger.info("Artifacts saved successfully")
    joblib.dump({"model_name": model_name, "model": model}, artifacts_dir / "model.joblib")


def train_pipeline(df_boot: pd.DataFrame, artifacts_dir: Path) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    logger.info("Train pipeline started | rows=%d", len(df_boot))
    # labels
    df = df_boot.copy()
    df["Spam"] = df.apply(lambda r: classify_email(r["Subject"], r["Cleaned_Body"]), axis=1)

    # word2vec
    tokenized = df["Combined_Text"].apply(lambda x: str(x).split())
    w2v = train_word2vec(tokenized)

    # vectors + split
    X = vectorize_texts(df["Combined_Text"], w2v)
    y = df["Spam"]
    spam_count = int(df["Spam"].sum())
    ham_count = len(df) - spam_count
    logger.info("Label distribution | spam=%d ham=%d", spam_count, ham_count)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    logger.info("Train/test split | X_train=%s X_test=%s",X_train.shape,X_test.shape,)
    models = train_models(X_train, y_train)

    # choose best by training accuracy quickly? (evaluation module will choose properly)
    # We'll save a default, evaluation can overwrite if needed.
    logger.info("Saving initial model before evaluation")
    save_artifacts(artifacts_dir, w2v, "LogisticRegression", models["LogisticRegression"])
    
    logger.info("Train pipeline completed")
    return X_train, X_test, y_train, y_test
