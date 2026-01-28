from pathlib import Path
from spam_classifier.predict import predict_text

def test_predict_smoke():
    artifacts = Path("artifacts")
    # if artifacts don't exist, this test will fail -> encourages you to train first in CI/CD
    assert (artifacts / "model.joblib").exists()
    assert (artifacts / "word2vec.model").exists()

    out = predict_text("free money urgent click now", artifacts)
    assert "label" in out
    assert out["label"] in {"Spam", "Ham"}

