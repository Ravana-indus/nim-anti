"""In-process telemetry for observability endpoints."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from statistics import mean
from typing import Deque


class TelemetryStore:
    """Lightweight in-memory metrics store for runtime diagnostics."""

    def __init__(self):
        self._lock = threading.Lock()
        self._http_total = 0
        self._http_by_status = defaultdict(int)
        self._http_by_path = defaultdict(int)
        self._http_latencies_ms: Deque[float] = deque(maxlen=5000)
        self._http_timestamps: Deque[float] = deque(maxlen=5000)

        self._provider_attempt_total = 0
        self._provider_attempt_by_status = defaultdict(int)
        self._provider_attempt_by_model = defaultdict(int)
        self._provider_errors_by_type = defaultdict(int)
        self._provider_latency_ms: Deque[float] = deque(maxlen=5000)
        self._provider_timestamps: Deque[float] = deque(maxlen=5000)

    def record_http(self, method: str, path: str, status_code: int, latency_ms: float) -> None:
        now = time.time()
        status_group = f"{int(status_code / 100)}xx"
        with self._lock:
            self._http_total += 1
            self._http_by_status[status_group] += 1
            self._http_by_path[f"{method} {path}"] += 1
            self._http_latencies_ms.append(max(0.0, latency_ms))
            self._http_timestamps.append(now)

    def record_provider_attempt(
        self,
        model: str,
        key_suffix: str,
        status: str,
        latency_ms: float,
        stream: bool,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        now = time.time()
        _ = key_suffix
        _ = stream
        _ = error_message
        with self._lock:
            self._provider_attempt_total += 1
            self._provider_attempt_by_status[status] += 1
            self._provider_attempt_by_model[model] += 1
            if error_type:
                self._provider_errors_by_type[error_type] += 1
            self._provider_latency_ms.append(max(0.0, latency_ms))
            self._provider_timestamps.append(now)

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int((len(sorted_values) - 1) * percentile)
        return sorted_values[index]

    @staticmethod
    def _rpm(timestamps: Deque[float]) -> int:
        now = time.time()
        minute_ago = now - 60
        return sum(1 for value in timestamps if value >= minute_ago)

    def snapshot(self) -> dict:
        with self._lock:
            http_latencies = list(self._http_latencies_ms)
            provider_latencies = list(self._provider_latency_ms)
            return {
                "http": {
                    "total_requests": self._http_total,
                    "requests_per_minute": self._rpm(self._http_timestamps),
                    "by_status": dict(self._http_by_status),
                    "by_path": dict(self._http_by_path),
                    "latency_ms_avg": round(mean(http_latencies), 2) if http_latencies else 0.0,
                    "latency_ms_p95": round(self._percentile(http_latencies, 0.95), 2),
                },
                "provider_attempts": {
                    "total": self._provider_attempt_total,
                    "attempts_per_minute": self._rpm(self._provider_timestamps),
                    "by_status": dict(self._provider_attempt_by_status),
                    "by_model": dict(self._provider_attempt_by_model),
                    "errors_by_type": dict(self._provider_errors_by_type),
                    "latency_ms_avg": round(mean(provider_latencies), 2)
                    if provider_latencies
                    else 0.0,
                    "latency_ms_p95": round(
                        self._percentile(provider_latencies, 0.95), 2
                    ),
                },
            }

    def as_prometheus(self) -> str:
        data = self.snapshot()
        lines = [
            "# HELP ccnim_http_requests_total Total HTTP requests",
            "# TYPE ccnim_http_requests_total counter",
            f"ccnim_http_requests_total {data['http']['total_requests']}",
            "# HELP ccnim_http_requests_per_minute HTTP requests in last 60s",
            "# TYPE ccnim_http_requests_per_minute gauge",
            f"ccnim_http_requests_per_minute {data['http']['requests_per_minute']}",
            "# HELP ccnim_http_latency_ms_avg Average HTTP latency in milliseconds",
            "# TYPE ccnim_http_latency_ms_avg gauge",
            f"ccnim_http_latency_ms_avg {data['http']['latency_ms_avg']}",
            "# HELP ccnim_http_latency_ms_p95 P95 HTTP latency in milliseconds",
            "# TYPE ccnim_http_latency_ms_p95 gauge",
            f"ccnim_http_latency_ms_p95 {data['http']['latency_ms_p95']}",
            "# HELP ccnim_provider_attempts_total Total upstream provider attempts",
            "# TYPE ccnim_provider_attempts_total counter",
            f"ccnim_provider_attempts_total {data['provider_attempts']['total']}",
            "# HELP ccnim_provider_attempts_per_minute Provider attempts in last 60s",
            "# TYPE ccnim_provider_attempts_per_minute gauge",
            f"ccnim_provider_attempts_per_minute {data['provider_attempts']['attempts_per_minute']}",
            "# HELP ccnim_provider_latency_ms_avg Average provider latency in milliseconds",
            "# TYPE ccnim_provider_latency_ms_avg gauge",
            f"ccnim_provider_latency_ms_avg {data['provider_attempts']['latency_ms_avg']}",
            "# HELP ccnim_provider_latency_ms_p95 P95 provider latency in milliseconds",
            "# TYPE ccnim_provider_latency_ms_p95 gauge",
            f"ccnim_provider_latency_ms_p95 {data['provider_attempts']['latency_ms_p95']}",
        ]
        return "\n".join(lines) + "\n"


telemetry = TelemetryStore()
