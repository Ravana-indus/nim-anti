"""Admin API routes for cc-nim management."""

import asyncio
import json
from pathlib import Path
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Deque, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from .dependencies import get_provider
from .telemetry import telemetry
from config.settings import (
    clear_active_model_override,
    get_active_model,
    get_settings,
    has_active_model_override,
    persist_model_to_env,
    set_active_model,
)

router = APIRouter(prefix="/admin", tags=["admin"])

# In-memory request log buffer (last 1000 requests).
request_logs: Deque[dict] = deque(maxlen=1000)


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        for connection in self.active_connections.copy():
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)


manager = ConnectionManager()


class ActiveModelUpdateRequest(BaseModel):
    model: str
    persist: bool = False


_model_catalog_cache: Optional[list[str]] = None


def _safe_get_provider() -> tuple[Optional[object], Optional[str]]:
    """Get provider safely without breaking admin dashboard rendering."""
    try:
        return get_provider(), None
    except Exception as exc:
        return None, str(exc)


def _load_model_catalog() -> list[str]:
    """Load and cache available NIM models from local catalog file."""
    global _model_catalog_cache
    if _model_catalog_cache is not None:
        return _model_catalog_cache

    catalog_path = Path("nvidia_nim_models.json")
    if not catalog_path.exists():
        _model_catalog_cache = []
        return _model_catalog_cache

    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        raw_models = payload.get("data", []) if isinstance(payload, dict) else []
        models: list[str] = []
        for item in raw_models:
            if isinstance(item, dict):
                model_id = item.get("id")
                if isinstance(model_id, str) and model_id.strip():
                    models.append(model_id.strip())
        _model_catalog_cache = sorted(set(models))
    except Exception:
        _model_catalog_cache = []

    return _model_catalog_cache


@dataclass
class RequestLogEntry:
    """Structure for a request log entry."""

    timestamp: str
    model: str
    key_suffix: str
    status: str
    response_time_ms: float
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def log_request(
    model: str,
    key: str,
    status: str,
    response_time_ms: float,
    error: Optional[str] = None,
):
    """Log a request to the in-memory buffer and broadcast to WebSocket."""
    entry = RequestLogEntry(
        timestamp=datetime.utcnow().isoformat() + "Z",
        model=model,
        key_suffix=key[-4:] if len(key) >= 4 else key,
        status=status,
        response_time_ms=response_time_ms,
        error=error,
    )
    event = entry.to_dict()
    request_logs.appendleft(event)
    broadcast_coro = manager.broadcast({"type": "log", "data": event})
    try:
        asyncio.create_task(broadcast_coro)
    except RuntimeError:
        # No active loop (e.g. sync test context) - skip WS broadcast.
        broadcast_coro.close()


async def _provider_snapshot(provider: object) -> dict:
    """Collect provider snapshot using public methods when available."""
    if provider is None:
        return {
            "keys": [],
            "health": {},
            "runtime": {},
        }

    if hasattr(provider, "get_admin_snapshot"):
        return await provider.get_admin_snapshot()

    key_manager = getattr(provider, "_key_manager", None)
    key_statuses: list[dict] = []
    if key_manager:
        if hasattr(key_manager, "snapshot"):
            key_statuses = await key_manager.snapshot()
        else:
            for key in key_manager.keys:
                key_statuses.append(
                    {
                        "key_suffix": key[-4:] if len(key) >= 4 else key,
                        "key_masked": f"{key[:8]}****{key[-4:]}",
                        "blocked": False,
                        "cooldown_remaining_seconds": 0.0,
                        "usage_count": 0,
                        "remaining_requests": 0,
                        "in_flight": 0,
                        "capacity_percent": 0.0,
                    }
                )

    return {
        "keys": key_statuses,
        "health": {},
        "runtime": {},
    }


def _aggregate_from_logs() -> dict:
    logs = list(request_logs)[:500]
    total = len(logs)
    successful = sum(1 for log in logs if log["status"] == "success")
    failed = total - successful
    telemetry_snapshot = telemetry.snapshot()
    return {
        "total_requests": total,
        "successful": successful,
        "failed": failed,
        "success_rate": round((successful / total * 100) if total > 0 else 100, 1),
        "requests_per_minute": telemetry_snapshot["http"]["requests_per_minute"],
        "avg_http_latency_ms": telemetry_snapshot["http"]["latency_ms_avg"],
        "p95_http_latency_ms": telemetry_snapshot["http"]["latency_ms_p95"],
    }


def _legacy_health_from_telemetry(active_model: str) -> dict:
    """Build a minimal legacy model-health payload for older admin UIs."""
    telemetry_snapshot = telemetry.snapshot()
    provider_meta = telemetry_snapshot.get("provider_attempts", {})
    by_model = provider_meta.get("by_model", {}) or {}
    total = int(by_model.get(active_model, 0))
    return {
        active_model: {
            "total": total,
            "success": total,
            "success_rate": 100.0 if total > 0 else 0.0,
            "recent_errors": [],
        }
    }


@router.get("/status")
async def get_status():
    """Get overall system status."""
    provider, provider_error = _safe_get_provider()
    settings = get_settings()
    snapshot = await _provider_snapshot(provider)

    active_model = get_active_model()
    health = snapshot.get("health", {})
    if not health:
        health = _legacy_health_from_telemetry(active_model)

    return {
        "active_model": active_model,
        "default_model": settings.model,
        "has_runtime_model_override": has_active_model_override(),
        # Backward compatibility for older dashboard JS.
        "model_chain": [active_model],
        "keys": snapshot.get("keys", []),
        "health": health,
        "runtime": snapshot.get("runtime", {}),
        "provider_error": provider_error,
        "aggregate": _aggregate_from_logs(),
        "settings": {
            "rate_limit": settings.nvidia_nim_rate_limit,
            "rate_window": settings.nvidia_nim_rate_window,
            "cooldown_seconds": settings.nvidia_nim_key_cooldown_seconds,
            "max_in_flight": settings.nvidia_nim_max_in_flight,
        },
    }


@router.get("/metrics")
async def get_metrics():
    """Get telemetry snapshot and provider runtime metrics."""
    provider, provider_error = _safe_get_provider()
    snapshot = await _provider_snapshot(provider)
    return {
        "telemetry": telemetry.snapshot(),
        "provider_runtime": snapshot.get("runtime", {}),
        "provider_error": provider_error,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@router.get("/keys")
async def get_keys():
    """Get detailed API key status."""
    provider, _ = _safe_get_provider()
    snapshot = await _provider_snapshot(provider)
    return {"keys": snapshot.get("keys", [])}


@router.get("/model")
async def get_model():
    """Get active model configuration."""
    settings = get_settings()
    return {
        "active_model": get_active_model(),
        "default_model": settings.model,
        "has_runtime_model_override": has_active_model_override(),
    }


@router.get("/models")
async def get_models(q: Optional[str] = None, limit: int = 200):
    """Get available NIM model catalog with optional search."""
    safe_limit = max(1, min(limit, 1000))
    models = _load_model_catalog()
    if q:
        needle = q.strip().lower()
        models = [model for model in models if needle in model.lower()]
    return {
        "models": models[:safe_limit],
        "total": len(models),
    }


@router.post("/model")
async def set_model(payload: ActiveModelUpdateRequest):
    """Set active runtime model for new requests."""
    try:
        model = set_active_model(payload.model)
        if payload.persist:
            persist_model_to_env(model)
        return {
            "status": "updated",
            "active_model": model,
            "has_runtime_model_override": True,
            "persisted": payload.persist,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/model/reset")
async def reset_model():
    """Reset model selection to default MODEL from settings."""
    clear_active_model_override()
    settings = get_settings()
    return {
        "status": "reset",
        "active_model": settings.model,
        "has_runtime_model_override": False,
    }


@router.get("/logs")
async def get_logs(
    model: Optional[str] = None,
    key_suffix: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
):
    """Get recent request logs (paginated/filtered)."""
    safe_limit = max(1, min(limit, 1000))
    logs = list(request_logs)

    if model:
        logs = [log for log in logs if log["model"] == model]
    if key_suffix:
        logs = [log for log in logs if log["key_suffix"] == key_suffix]
    if status:
        logs = [log for log in logs if log["status"] == status]

    return {"logs": logs[:safe_limit]}


@router.post("/keys/{key_suffix}/block")
async def block_key(key_suffix: str):
    """Block a specific API key."""
    provider, provider_error = _safe_get_provider()
    if provider is None:
        raise HTTPException(
            status_code=503,
            detail=f"Provider not available: {provider_error or 'initialization failed'}",
        )
    key_manager = getattr(provider, "_key_manager", None)
    if not key_manager:
        raise HTTPException(status_code=500, detail="Key manager not available")

    target_key = key_manager.find_key_by_suffix(key_suffix)
    if not target_key:
        raise HTTPException(status_code=404, detail="Key not found")

    await key_manager.set_manual_block(target_key, 86400)
    return {"status": "blocked", "key_suffix": key_suffix}


@router.post("/keys/{key_suffix}/unblock")
async def unblock_key(key_suffix: str):
    """Unblock a specific API key."""
    provider, provider_error = _safe_get_provider()
    if provider is None:
        raise HTTPException(
            status_code=503,
            detail=f"Provider not available: {provider_error or 'initialization failed'}",
        )
    key_manager = getattr(provider, "_key_manager", None)
    if not key_manager:
        raise HTTPException(status_code=500, detail="Key manager not available")

    target_key = key_manager.find_key_by_suffix(key_suffix)
    if not target_key:
        raise HTTPException(status_code=404, detail="Key not found")

    await key_manager.clear_manual_block(target_key)
    return {"status": "unblocked", "key_suffix": key_suffix}


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time request logs."""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@router.get("/")
async def admin_index():
    """Serve the admin dashboard."""
    from fastapi.responses import FileResponse

    return FileResponse("static/admin/index.html")
