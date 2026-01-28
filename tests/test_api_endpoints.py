import os
from fastapi.testclient import TestClient

from spam_classifier.api import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_stats():
    r = client.get("/stats")
    assert r.status_code == 200
    assert "total_requests" in r.json()

def test_metrics_exists_or_404():
    r = client.get("/metrics")
    assert r.status_code in (200, 404)  # 404 is acceptable before training

def test_model_info_exists_or_404():
    r = client.get("/model/info")
    assert r.status_code in (200, 404)

def test_reload_requires_admin():
    r = client.post("/admin/reload")
    assert r.status_code in (401, 403)
