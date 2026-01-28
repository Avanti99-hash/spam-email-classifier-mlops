from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def evaluate_and_select_best(
    X_train: np.ndarray,
    y_train,
    X_test: np.ndarray,
    y_test,
) -> Tuple[str, object, Dict]:
    models = {
        "RandomForest": RandomForestClassifier(random_state=42, n_estimators=100),
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
        "SVM": SVC(random_state=42, probability=True),
    }

    results = []
    best = ("", None, -1.0, None)

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, pred))
        report = classification_report(y_test, pred, output_dict=True)

        metrics = {
            "accuracy": acc,
            "precision_weighted": float(report["weighted avg"]["precision"]),
            "recall_weighted": float(report["weighted avg"]["recall"]),
            "f1_weighted": float(report["weighted avg"]["f1-score"]),
        }
        results.append({"model": name, **metrics})

        if acc > best[2]:
            best = (name, model, acc, metrics)

    best_name, best_model, best_acc, best_metrics = best
    payload = {"best_model": best_name, "best_metrics": best_metrics, "all_models": results}
    return best_name, best_model, payload


def save_metrics(artifacts_dir: Path, payload: Dict) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def quality_gate(payload: Dict, min_accuracy: float = 0.90) -> None:
    acc = payload["best_metrics"]["accuracy"]
    if acc < min_accuracy:
        raise SystemExit(f"Quality gate failed: accuracy {acc:.4f} < {min_accuracy:.2f}")


def save_best_model(artifacts_dir: Path, best_name: str, best_model) -> None:
    obj = joblib.load(artifacts_dir / "model.joblib")
    joblib.dump({"model_name": best_name, "model": best_model}, artifacts_dir / "model.joblib")


def run_evaluation(
    artifacts_dir: Path,
    X_train,
    y_train,
    X_test,
    y_test,
    min_accuracy: float = 0.90,
) -> Dict:
    best_name, best_model, payload = evaluate_and_select_best(X_train, y_train, X_test, y_test)
    save_metrics(artifacts_dir, payload)
    quality_gate(payload, min_accuracy=min_accuracy)
    save_best_model(artifacts_dir, best_name, best_model)
    return payload
