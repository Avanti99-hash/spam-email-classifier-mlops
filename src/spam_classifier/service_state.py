from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Stats:
    total_requests: int = 0
    predict_requests: int = 0
    batch_requests: int = 0
    spam_predictions: int = 0
    ham_predictions: int = 0
    errors: int = 0
    total_latency_ms: float = 0.0
    last_reload_ts: float | None = None
    started_ts: float = field(default_factory=time.time)

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_request(self, endpoint: str, latency_ms: float, predicted_label: str | None = None):
        with self._lock:
            self.total_requests += 1
            self.total_latency_ms += latency_ms

            if endpoint == "predict":
                self.predict_requests += 1
            elif endpoint == "predict_batch":
                self.batch_requests += 1

            if predicted_label == "Spam":
                self.spam_predictions += 1
            elif predicted_label == "Ham":
                self.ham_predictions += 1

    def record_error(self):
        with self._lock:
            self.errors += 1

    def mark_reload(self):
        with self._lock:
            self.last_reload_ts = time.time()

    def snapshot(self) -> Dict:
        with self._lock:
            avg = (self.total_latency_ms / self.total_requests) if self.total_requests else 0.0
            uptime_s = time.time() - self.started_ts
            return {
                "uptime_seconds": round(uptime_s, 2),
                "total_requests": self.total_requests,
                "predict_requests": self.predict_requests,
                "batch_requests": self.batch_requests,
                "spam_predictions": self.spam_predictions,
                "ham_predictions": self.ham_predictions,
                "errors": self.errors,
                "avg_latency_ms": round(avg, 2),
                "last_reload_ts": self.last_reload_ts,
            }
