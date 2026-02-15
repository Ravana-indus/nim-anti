from unittest.mock import patch

from fastapi.testclient import TestClient

from api.app import app
from api.admin import log_request, request_logs


class _MockAdminProvider:
    async def get_admin_snapshot(self):
        return {
            "keys": [
                {
                    "key_suffix": "1234",
                    "key_masked": "nvapi-ke****1234",
                    "blocked": False,
                    "cooldown_remaining_seconds": 0.0,
                    "usage_count": 2,
                    "remaining_requests": 38,
                    "in_flight": 1,
                    "capacity_percent": 95.0,
                }
            ],
            "health": {
                "model-primary": {
                    "total": 10,
                    "success": 9,
                    "success_rate": 90.0,
                    "recent_errors": [],
                }
            },
            "runtime": {
                "active_requests": 1,
                "max_in_flight": 32,
                "client_pool_size": 2,
            },
        }


def test_admin_status_masks_keys_and_includes_runtime():
    client = TestClient(app)
    with (
        patch("api.admin.get_provider", return_value=_MockAdminProvider()),
        patch("api.admin.get_active_model", return_value="runtime-model"),
        patch("api.admin.has_active_model_override", return_value=True),
    ):
        response = client.get("/admin/status")

    assert response.status_code == 200
    data = response.json()
    assert data["keys"][0]["key_masked"] == "nvapi-ke****1234"
    assert "key_full" not in data["keys"][0]
    assert data["runtime"]["max_in_flight"] == 32
    assert data["active_model"] == "runtime-model"
    assert data["model_chain"][0] == "runtime-model"
    assert "quick_switch_models" in data


def test_admin_status_graceful_when_provider_init_fails():
    client = TestClient(app)
    with (
        patch("api.admin.get_provider", side_effect=RuntimeError("provider init failed")),
        patch("api.admin.get_active_model", return_value="runtime-model"),
        patch("api.admin.has_active_model_override", return_value=True),
    ):
        response = client.get("/admin/status")

    assert response.status_code == 200
    data = response.json()
    assert data["keys"] == []
    assert "provider init failed" in data["provider_error"]


def test_admin_metrics_endpoint():
    client = TestClient(app)
    with patch("api.admin.get_provider", return_value=_MockAdminProvider()):
        response = client.get("/admin/metrics")

    assert response.status_code == 200
    payload = response.json()
    assert "telemetry" in payload
    assert payload["provider_runtime"]["client_pool_size"] == 2


def test_admin_metrics_graceful_when_provider_init_fails():
    client = TestClient(app)
    with patch("api.admin.get_provider", side_effect=RuntimeError("provider init failed")):
        response = client.get("/admin/metrics")

    assert response.status_code == 200
    payload = response.json()
    assert "telemetry" in payload
    assert payload["provider_runtime"] == {}
    assert "provider init failed" in payload["provider_error"]


def test_metrics_prometheus_endpoint():
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "ccnim_http_requests_total" in response.text


def test_admin_logs_limit_is_clamped():
    client = TestClient(app)
    log_request("test-model", "test-key", "success", 10.0)
    response = client.get("/admin/logs?limit=50000")
    assert response.status_code == 200
    assert len(response.json()["logs"]) <= 1000


def test_admin_set_model_endpoint():
    client = TestClient(app)
    sticky_provider = type(
        "StickyProviderMock", (), {"set_sticky_model": lambda self, model: None}
    )()
    with patch("api.admin.set_active_model", return_value="new-model"), patch(
        "api.admin.get_provider", return_value=sticky_provider
    ):
        response = client.post("/admin/model", json={"model": "new-model"})

    assert response.status_code == 200
    assert response.json()["active_model"] == "new-model"
    assert response.json()["persisted"] is False


def test_admin_set_model_with_persist():
    client = TestClient(app)
    with patch("api.admin.set_active_model", return_value="new-model"), patch(
        "api.admin.persist_model_to_env"
    ) as mock_persist:
        response = client.post(
            "/admin/model", json={"model": "new-model", "persist": True}
        )

    assert response.status_code == 200
    assert response.json()["persisted"] is True
    mock_persist.assert_called_once_with("new-model")


def test_admin_reset_model_endpoint():
    client = TestClient(app)
    sticky_provider = type(
        "StickyProviderMock", (), {"clear_sticky_model": lambda self: None}
    )()
    with patch("api.admin.clear_active_model_override"), patch(
        "api.admin.get_settings"
    ) as mock_settings, patch("api.admin.get_provider", return_value=sticky_provider):
        mock_settings.return_value.model = "default-from-env"
        response = client.post("/admin/model/reset")

    assert response.status_code == 200
    assert response.json()["active_model"] == "default-from-env"


def test_admin_set_model_syncs_provider_sticky_model():
    client = TestClient(app)

    class _Provider:
        def __init__(self):
            self.model = None

        def set_sticky_model(self, model: str):
            self.model = model

    provider = _Provider()
    with patch("api.admin.set_active_model", return_value="new-model"), patch(
        "api.admin.get_provider", return_value=provider
    ):
        response = client.post("/admin/model", json={"model": "new-model"})

    assert response.status_code == 200
    assert provider.model == "new-model"


def test_admin_reset_model_clears_provider_sticky_model():
    client = TestClient(app)

    class _Provider:
        def __init__(self):
            self.cleared = False

        def clear_sticky_model(self):
            self.cleared = True

    provider = _Provider()
    with patch("api.admin.clear_active_model_override"), patch(
        "api.admin.get_settings"
    ) as mock_settings, patch("api.admin.get_provider", return_value=provider):
        mock_settings.return_value.model = "default-from-env"
        response = client.post("/admin/model/reset")

    assert response.status_code == 200
    assert provider.cleared is True


def test_admin_model_catalog_endpoint():
    client = TestClient(app)
    response = client.get("/admin/models?limit=50")
    assert response.status_code == 200
    payload = response.json()
    assert "models" in payload
    assert "total" in payload
    assert len(payload["models"]) <= 50


def test_model_performance_avg_latency_uses_success_only():
    client = TestClient(app)
    request_logs.clear()
    request_logs.appendleft(
        {
            "timestamp": "2026-02-15T00:00:00Z",
            "model": "m1",
            "key_suffix": "1111",
            "status": "success",
            "response_time_ms": 1000.0,
            "error": None,
        }
    )
    request_logs.appendleft(
        {
            "timestamp": "2026-02-15T00:00:01Z",
            "model": "m1",
            "key_suffix": "1111",
            "status": "failed",
            "response_time_ms": 100000.0,
            "error": "bad request",
        }
    )
    request_logs.appendleft(
        {
            "timestamp": "2026-02-15T00:00:02Z",
            "model": "m1",
            "key_suffix": "1111",
            "status": "fallback",
            "response_time_ms": 2000.0,
            "error": "Switching to m2",
        }
    )

    response = client.get("/admin/models/performance")
    assert response.status_code == 200
    model = response.json()["models"][0]
    assert model["model"] == "m1"
    assert model["avg_latency_ms"] == 1000.0
    assert model["avg_attempt_latency_ms"] == round((1000.0 + 100000.0 + 2000.0) / 3, 2)
