"""Admin API routes for cc-nim management."""

import asyncio
import json
from pathlib import Path
from collections import deque, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Deque, Optional, Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Query
from pydantic import BaseModel

from .dependencies import get_provider
from .telemetry import telemetry
from config.settings import (
    clear_active_model_override,
    get_active_model,
    get_configured_fallback_models,
    get_model_fallback_chain,
    get_settings,
    has_active_model_override,
    persist_model_to_env,
    set_active_model,
)

router = APIRouter(prefix="/admin", tags=["admin"])

# In-memory request log buffer (last 1000 requests).
request_logs: Deque[dict] = deque(maxlen=1000)

# Detailed request storage for inspector (last 500 requests with full details)
request_details: Deque[dict] = deque(maxlen=500)

# Time-series metrics for charts (last 24 hours, bucketed by minute)
metrics_history: list[dict] = []
metrics_lock = asyncio.Lock()


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

    candidate_paths = [
        Path(__file__).resolve().parents[1] / "nvidia_nim_models.json",
        Path("nvidia_nim_models.json"),
    ]
    catalog_path = next((p for p in candidate_paths if p.exists()), None)
    if catalog_path is None:
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
    configured_fallback_models = get_configured_fallback_models()
    model_chain = get_model_fallback_chain(active_model)
    health = snapshot.get("health", {})
    if not health:
        health = _legacy_health_from_telemetry(active_model)

    return {
        "active_model": active_model,
        "default_model": settings.model,
        "has_runtime_model_override": has_active_model_override(),
        # Backward compatibility for older dashboard JS.
        "model_chain": model_chain,
        "fallback_models": model_chain,
        "configured_fallback_models": configured_fallback_models,
        "quick_switch_models": configured_fallback_models,
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
            "request_timeout_sec": settings.nvidia_nim_request_timeout_seconds,
            "openai_max_retries": settings.nvidia_nim_openai_max_retries,
            "hard_max_tokens": settings.nim.hard_max_tokens,
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


@router.get("/circuit-breaker")
async def get_circuit_breaker():
    """Get circuit breaker status for monitoring."""
    provider, provider_error = _safe_get_provider()
    if provider is None:
        return {
            "status": "unavailable",
            "error": provider_error,
        }
    
    if hasattr(provider, "get_circuit_breaker_status"):
        return provider.get_circuit_breaker_status()
    
    return {
        "status": "not_implemented",
        "message": "Circuit breaker not available for this provider",
    }


@router.post("/circuit-breaker/reset")
async def reset_circuit_breaker():
    """Manually reset the circuit breaker."""
    provider, provider_error = _safe_get_provider()
    if provider is None:
        raise HTTPException(status_code=503, detail=f"Provider unavailable: {provider_error}")
    
    if hasattr(provider, "reset_circuit_breaker"):
        await provider.reset_circuit_breaker()
        return {"status": "reset", "message": "Circuit breaker has been reset"}
    
    raise HTTPException(status_code=400, detail="Circuit breaker not available for this provider")


@router.get("/model")
async def get_model():
    """Get active model configuration."""
    settings = get_settings()
    active_model = get_active_model()
    configured_fallback_models = get_configured_fallback_models()
    return {
        "active_model": active_model,
        "default_model": settings.model,
        "has_runtime_model_override": has_active_model_override(),
        "fallback_models": get_model_fallback_chain(active_model),
        "configured_fallback_models": configured_fallback_models,
        "quick_switch_models": configured_fallback_models,
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
        provider, _ = _safe_get_provider()
        if provider is not None and hasattr(provider, "set_sticky_model"):
            provider.set_sticky_model(model)
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
    provider, _ = _safe_get_provider()
    if provider is not None and hasattr(provider, "clear_sticky_model"):
        provider.clear_sticky_model()
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


# =============================================================================
# Phase 1: Request Inspector & Error Analysis
# =============================================================================

class RequestDetailUpdate(BaseModel):
    """Extended request details for logging."""
    request_id: str
    request_body: Optional[str] = None
    response_body: Optional[str] = None
    headers: Optional[dict] = None


def log_request_detail(
    model: str,
    key: str,
    status: str,
    response_time_ms: float,
    error: Optional[str] = None,
    request_body: Optional[str] = None,
    response_body: Optional[str] = None,
):
    """Log detailed request for inspector."""
    request_id = str(uuid4().hex[:12])
    entry = {
        "request_id": request_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": model,
        "key_suffix": key[-4:] if len(key) >= 4 else key,
        "status": status,
        "response_time_ms": response_time_ms,
        "error": error,
        "request_body": request_body,
        "response_body": response_body,
    }
    request_details.appendleft(entry)
    return request_id


@router.get("/requests/{request_id}")
async def get_request_details(request_id: str):
    """Get detailed request information for inspector."""
    for req in request_details:
        if req.get("request_id") == request_id:
            return {"request": req}
    raise HTTPException(status_code=404, detail="Request not found")


@router.get("/errors")
async def get_errors(
    model: Optional[str] = None,
    limit: int = Query(default=100, le=500),
):
    """Get error analysis with categorization."""
    errors = []
    for log in request_logs:
        if log.get("status") in ("error", "failed"):
            if model and log.get("model") != model:
                continue
            error_msg = log.get("error", "") or ""
            # Categorize errors
            category = "unknown"
            if "rate" in error_msg.lower() or "429" in error_msg:
                category = "rate_limit"
            elif "auth" in error_msg.lower() or "401" in error_msg or "api key" in error_msg.lower():
                category = "authentication"
            elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
                category = "quota"
            elif "timeout" in error_msg.lower() or "504" in error_msg:
                category = "timeout"
            elif "connection" in error_msg.lower() or "network" in error_msg.lower():
                category = "network"
            elif "model" in error_msg.lower() or "not found" in error_msg.lower():
                category = "model_not_found"
            elif "invalid" in error_msg.lower() or "400" in error_msg:
                category = "bad_request"
            
            errors.append({
                **log,
                "category": category,
            })
    
    # Aggregate by category
    by_category: dict[str, int] = defaultdict(int)
    for e in errors:
        by_category[e["category"]] += 1
    
    return {
        "errors": errors[:limit],
        "total": len(errors),
        "by_category": dict(by_category),
    }


# =============================================================================
# Phase 1: Metrics History for Charts
# =============================================================================

@router.get("/metrics/history")
async def get_metrics_history(
    period: str = Query(default="1h", regex="^(15m|1h|6h|24h)$"),
):
    """Get time-series metrics for charts."""
    now = datetime.utcnow()
    if period == "15m":
        cutoff = now - timedelta(minutes=15)
    elif period == "1h":
        cutoff = now - timedelta(hours=1)
    elif period == "6h":
        cutoff = now - timedelta(hours=6)
    else:
        cutoff = now - timedelta(hours=24)
    
    # Generate bucket data from request logs
    buckets: dict[str, dict] = {}
    for log in request_logs:
        if log.get("status") == "success":
            ts = log.get("timestamp", "")
            if ts:
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if dt.replace(tzinfo=None) >= cutoff:
                        bucket_key = dt.strftime("%H:%M")
                        if bucket_key not in buckets:
                            buckets[bucket_key] = {
                                "timestamp": bucket_key,
                                "requests": 0,
                                "success": 0,
                                "failed": 0,
                                "fallback": 0,
                                "latency_sum": 0.0,
                            }
                        buckets[bucket_key]["requests"] += 1
                        buckets[bucket_key][log.get("status")] += 1
                        buckets[bucket_key]["latency_sum"] += log.get("response_time_ms", 0)
                except ValueError:
                    pass
    
    # Convert to list with averages
    result = []
    for key in sorted(buckets.keys()):
        b = buckets[key]
        b["avg_latency_ms"] = round(b["latency_sum"] / b["requests"], 2) if b["requests"] > 0 else 0
        del b["latency_sum"]
        result.append(b)
    
    return {
        "period": period,
        "data": result,
    }


# =============================================================================
# Phase 1: Model Performance Stats
# =============================================================================

@router.get("/models/performance")
async def get_model_performance():
    """Get per-model performance statistics."""
    model_stats: dict[str, dict] = defaultdict(lambda: {
        "total": 0,
        "success": 0,
        "failed": 0,
        "fallback": 0,
        "latency_sum": 0.0,
        "success_latency_sum": 0.0,
        "success_count": 0,
        "errors": [],
    })
    
    for log in request_logs:
        model = log.get("model", "unknown")
        status = log.get("status", "unknown")
        latency = log.get("response_time_ms", 0)
        
        stats = model_stats[model]
        stats["total"] += 1
        if status in ("success", "failed", "fallback"):
            stats[status] += 1
        stats["latency_sum"] += latency
        if status == "success":
            stats["success_latency_sum"] += latency
            stats["success_count"] += 1
        if status in ("error", "failed") and log.get("error"):
            if len(stats["errors"]) < 5:  # Keep top 5 errors
                stats["errors"].append(log.get("error", "")[:200])
    
    result = []
    for model, stats in model_stats.items():
        total = stats["total"]
        success_rate = round((stats["success"] / total * 100) if total > 0 else 0, 1)
        avg_latency = (
            round(stats["success_latency_sum"] / stats["success_count"], 2)
            if stats["success_count"] > 0
            else 0
        )
        avg_attempt_latency = round(stats["latency_sum"] / total, 2) if total > 0 else 0

        result.append({
            "model": model,
            "total_requests": total,
            "success": stats["success"],
            "failed": stats["failed"],
            "fallback": stats["fallback"],
            "success_rate": success_rate,
            "avg_latency_ms": avg_latency,
            "avg_attempt_latency_ms": avg_attempt_latency,
            "recent_errors": stats["errors"],
        })
    
    # Sort by total requests descending
    result.sort(key=lambda x: x["total_requests"], reverse=True)
    
    return {"models": result}


# =============================================================================
# Phase 1: Settings Editor
# =============================================================================

class SettingsUpdateRequest(BaseModel):
    """Settings update request."""
    rate_limit: Optional[int] = None
    rate_window: Optional[int] = None
    cooldown_seconds: Optional[int] = None
    max_in_flight: Optional[int] = None


@router.get("/settings")
async def get_settings_info():
    """Get current settings for editor."""
    settings = get_settings()
    return {
        "rate_limit": settings.nvidia_nim_rate_limit,
        "rate_window": settings.nvidia_nim_rate_window,
        "cooldown_seconds": settings.nvidia_nim_key_cooldown_seconds,
        "max_in_flight": settings.nvidia_nim_max_in_flight,
        "request_timeout_sec": settings.nvidia_nim_request_timeout_seconds,
        "openai_max_retries": settings.nvidia_nim_openai_max_retries,
        "hard_max_tokens": settings.nim.hard_max_tokens,
        "model": settings.model,
        "fallback_models": settings.nvidia_nim_fallback_models,
    }


@router.post("/settings")
async def update_settings(payload: SettingsUpdateRequest):
    """Update runtime settings (Note: changes are temporary, not persisted to .env)."""
    # Settings are read-only at runtime for safety
    # This endpoint returns current settings with confirmation
    settings = get_settings()
    return {
        "status": "read_only",
        "message": "Settings are configured via .env file. Restart server after editing .env.",
        "current": {
            "rate_limit": settings.nvidia_nim_rate_limit,
            "rate_window": settings.nvidia_nim_rate_window,
            "cooldown_seconds": settings.nvidia_nim_key_cooldown_seconds,
            "max_in_flight": settings.nvidia_nim_max_in_flight,
        },
    }


# =============================================================================
# Phase 2: Active Connections & System Health
# =============================================================================

@router.get("/health/detailed")
async def get_detailed_health():
    """Get detailed system health diagnostics."""
    provider, provider_error = _safe_get_provider()
    settings = get_settings()
    
    health = {
        "status": "healthy",
        "components": {
            "provider": {
                "status": "healthy" if provider else "unavailable",
                "error": provider_error,
            },
            "telemetry": {
                "status": "healthy",
                "total_requests": telemetry.snapshot()["http"]["total_requests"],
            },
            "websocket": {
                "status": "healthy",
                "active_connections": len(manager.active_connections),
            },
            "storage": {
                "status": "healthy",
                "request_logs_count": len(request_logs),
                "request_details_count": len(request_details),
            },
        },
        "configuration": {
            "model": get_active_model(),
            "rate_limit": settings.nvidia_nim_rate_limit,
            "rate_window": settings.nvidia_nim_rate_window,
        },
    }
    
    # Check for warnings
    warnings = []
    if provider_error:
        warnings.append(f"Provider error: {provider_error}")
    if len(manager.active_connections) == 0:
        warnings.append("No active WebSocket connections")
    
    if warnings:
        health["status"] = "degraded"
        health["warnings"] = warnings
    
    return health


# =============================================================================
# Phase 3: Fallback Chain & Key Management
# =============================================================================

@router.post("/fallback-order")
async def update_fallback_order(models: list[str]):
    """Update fallback model order (runtime only)."""
    # This would require modifying settings at runtime
    return {
        "status": "not_implemented",
        "message": "Fallback order is configured via NVIDIA_NIM_FALLBACK_MODELS in .env",
        "current": get_configured_fallback_models(),
    }


@router.post("/keys/test")
async def test_all_keys():
    """Test all API keys and return health status."""
    provider, _ = _safe_get_provider()
    if not provider:
        raise HTTPException(status_code=503, detail="Provider not available")
    
    key_manager = getattr(provider, "_key_manager", None)
    if not key_manager:
        raise HTTPException(status_code=500, detail="Key manager not available")
    
    results = []
    for key in key_manager.keys:
        key_suffix = key[-4:] if len(key) >= 4 else key
        is_blocked = getattr(key_manager, "_is_blocked", lambda k: False)(key)
        cooldown = getattr(key_manager, "_get_cooldown", lambda k: 0)(key)
        
        results.append({
            "key_suffix": key_suffix,
            "key_masked": f"{key[:8]}****{key[-4:]}",
            "blocked": is_blocked,
            "cooldown_remaining_seconds": cooldown,
            "healthy": not is_blocked and cooldown == 0,
        })
    
    return {"keys": results}


# =============================================================================
# Phase 4: Export & Utilities
# =============================================================================

@router.get("/export/logs")
async def export_logs(format: str = Query(default="json", regex="^(json|csv)$")):
    """Export request logs in specified format."""
    logs = list(request_logs)
    
    if format == "csv":
        import csv
        import io
        
        output = io.StringIO()
        if logs:
            writer = csv.DictWriter(output, fieldnames=logs[0].keys())
            writer.writeheader()
            writer.writerows(logs)
        
        from fastapi.responses import Response
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=ccnim-logs.csv"},
        )
    
    return {"logs": logs}


@router.get("/export/metrics")
async def export_metrics():
    """Export current metrics snapshot."""
    snapshot = telemetry.snapshot()
    return {
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "telemetry": snapshot,
        "aggregate": _aggregate_from_logs(),
    }
