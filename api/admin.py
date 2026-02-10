"""Admin API routes for cc-nim management."""

import asyncio
import json
import logging
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import AsyncGenerator, Deque, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from .dependencies import get_provider
from config.settings import Settings, get_settings
from providers.base import BaseProvider

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])

# In-memory request log buffer (last 1000 requests)
request_logs: Deque[dict] = deque(maxlen=1000)

# WebSocket connection manager
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

# Open access - no authentication required


@asynccontextmanager
async def get_lifecycle_events():
    """Context manager for lifecycle events (placeholder)."""
    yield


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
    request_logs.appendleft(entry.to_dict())
    
    # Broadcast to WebSocket clients
    asyncio.create_task(manager.broadcast({"type": "log", "data": entry.to_dict()}))


# --- Dashboard Endpoints (Open Access) ---

@router.get("/status")
async def get_status():
    """Get overall system status."""
    provider = get_provider()
    settings = get_settings()
    
    # Get key statuses from key manager
    key_manager = getattr(provider, '_key_manager', None)
    model_router = getattr(provider, '_model_router', None)
    
    keys_status = []
    if key_manager:
        now = time.monotonic()
        for key in key_manager.keys:
            key_status = {
                "key": key[:8] + "..." + key[-4:],
                "key_full": key,  # Full key for admin (masked in UI)
                "blocked": key_manager._cooldown_until.get(key, 0) > now,
                "remaining_requests": key_manager._rate_limit - len(key_manager._requests.get(key, [])),
                "in_flight": key_manager._in_flight.get(key, 0),
            }
            keys_status.append(key_status)
    
    model_chain = []
    if model_router:
        model_chain = model_router.get_model_chain()
    
    health_summary = {}
    if model_router:
        for model in model_router._health:
            h = model_router._health[model]
            health_summary[model] = {
                "total": h.total,
                "success": h.success,
                "success_rate": round(h.success_rate * 100, 1),
                "recent_errors": list(h.recent_errors)[-5:],  # Last 5 errors
            }
    
    # Calculate aggregate stats from logs
    recent_logs = list(request_logs)[:100]
    total_requests = len(recent_logs)
    successful = sum(1 for log in recent_logs if log["status"] == "success")
    failed = total_requests - successful
    
    # Calculate requests per minute (approximate from recent logs)
    now = datetime.utcnow()
    minute_ago = now.timestamp() - 60
    rpm = sum(1 for log in recent_logs if datetime.fromisoformat(log["timestamp"].replace("Z", "+00:00")).timestamp() > minute_ago)
    
    return {
        "model_chain": model_chain,
        "keys": keys_status,
        "health": health_summary,
        "aggregate": {
            "total_requests": total_requests,
            "successful": successful,
            "failed": failed,
            "success_rate": round((successful / total_requests * 100) if total_requests > 0 else 100, 1),
            "requests_per_minute": rpm,
        },
        "settings": {
            "rate_limit": settings.nvidia_nim_rate_limit,
            "rate_window": settings.nvidia_nim_rate_window,
            "cooldown_seconds": settings.nvidia_nim_key_cooldown_seconds,
        }
    }


@router.get("/keys")
async def get_keys():
    """Get detailed API key status."""
    provider = get_provider()
    key_manager = getattr(provider, '_key_manager', None)
    
    if not key_manager:
        return {"keys": []}
    
    now = time.monotonic()
    keys = []
    for key in key_manager.keys:
        # Get recent errors for this key (from executor logs would be better, but this is a proxy)
        keys.append({
            "key_masked": key[:8] + "****" + key[-4:],
            "key_suffix": key[-4:],
            "blocked": key_manager._cooldown_until.get(key, 0) > now,
            "cooldown_until": key_manager._cooldown_until.get(key, 0),
            "in_flight": key_manager._in_flight.get(key, 0),
            "usage_count": len(key_manager._requests.get(key, [])),
        })
    
    return {"keys": keys}


@router.get("/chain")
async def get_model_chain():
    """Get current model chain."""
    provider = get_provider()
    model_router = getattr(provider, '_model_router', None)
    
    if not model_router:
        return {"chain": []}
    
    return {
        "chain": model_router.get_model_chain(),
        "health": {
            model: {
                "total": h.total,
                "success": h.success,
                "success_rate": round(h.success_rate * 100, 1),
            }
            for model, h in model_router._health.items()
        }
    }


@router.get("/logs")
async def get_logs(
    model: Optional[str] = None,
    key_suffix: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
):
    """Get recent request logs (paginated/filtered)."""
    logs = list(request_logs)
    
    if model:
        logs = [log for log in logs if log["model"] == model]
    if key_suffix:
        logs = [log for log in logs if log["key_suffix"] == key_suffix]
    if status:
        logs = [log for log in logs if log["status"] == status]
    
    return {"logs": logs[:limit]}


@router.post("/keys/{key_suffix}/block")
async def block_key(key_suffix: str):
    """Block a specific API key."""
    provider = get_provider()
    key_manager = getattr(provider, '_key_manager', None)
    
    if not key_manager:
        raise HTTPException(status_code=500, detail="Key manager not available")
    
    # Find the key
    target_key = None
    for key in key_manager.keys:
        if key.endswith(key_suffix):
            target_key = key
            break
    
    if not target_key:
        raise HTTPException(status_code=404, detail="Key not found")
    
    # Set cooldown for a long time (effectively blocking)
    key_manager._cooldown_until[target_key] = time.monotonic() + 86400  # 24 hours
    
    return {"status": "blocked", "key": key_suffix}


@router.post("/keys/{key_suffix}/unblock")
async def unblock_key(key_suffix: str):
    """Unblock a specific API key."""
    provider = get_provider()
    key_manager = getattr(provider, '_key_manager', None)
    
    if not key_manager:
        raise HTTPException(status_code=500, detail="Key manager not available")
    
    # Find the key
    target_key = None
    for key in key_manager.keys:
        if key.endswith(key_suffix):
            target_key = key
            break
    
    if not target_key:
        raise HTTPException(status_code=404, detail="Key not found")
    
    # Remove cooldown
    key_manager._cooldown_until[target_key] = 0
    
    return {"status": "unblocked", "key": key_suffix}


@router.post("/chain/reorder")
async def reorder_chain(new_order: list[str]):
    """Reorder the model chain."""
    provider = get_provider()
    model_router = getattr(provider, '_model_router', None)
    
    if not model_router:
        raise HTTPException(status_code=500, detail="Model router not available")
    
    # Verify all models are valid
    valid_models = set(model_router._chain)
    requested_models = set(new_order)
    
    if requested_models != valid_models:
        raise HTTPException(
            status_code=400,
            detail="Invalid model chain. Must contain exactly the same models."
        )
    
    model_router._chain = new_order
    logger.info(f"Model chain reordered: {new_order}")
    
    return {"status": "reordered", "chain": new_order}


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time request logs."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle incoming messages
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
