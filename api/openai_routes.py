"""OpenAI-compatible raw passthrough endpoints.

Uses raw httpx streaming to relay SSE bytes directly from NVIDIA NIM
to the client. Zero SDK deserialization overhead — just bytes in, bytes out.
"""

import logging
import time

try:
    import orjson

    def _json_bytes(obj) -> bytes:
        return orjson.dumps(obj)

    def _json_loads(data):
        return orjson.loads(data)
except ImportError:
    import json

    def _json_bytes(obj) -> bytes:
        return json.dumps(obj).encode()

    def _json_loads(data):
        return json.loads(data)

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse, Response

from .dependencies import get_provider, get_settings
from .telemetry import telemetry
from config.settings import Settings, get_active_model, get_configured_fallback_models
from providers.base import BaseProvider
from providers.exceptions import ProviderError

def _admin_log_request(**kwargs):
    """Lazy import to avoid circular dependencies with admin module."""
    try:
        from .admin import log_request
        return log_request(**kwargs)
    except ImportError:
        return None

logger = logging.getLogger(__name__)

router = APIRouter(tags=["openai"])


def _log_admin(model: str, key: str, status: str, ms: float, error: str = None):
    """Log to admin dashboard (lazy import to avoid circular deps)."""
    try:
        from api.admin import log_request
        log_request(model=model, key=key, status=status, response_time_ms=ms, error=error)
    except Exception:
        pass


# =============================================================================
# /v1/chat/completions  — raw httpx passthrough (zero SDK overhead)
# =============================================================================


@router.post("/v1/chat/completions")
async def chat_completions(
    raw_request: Request,
    provider: BaseProvider = Depends(get_provider),
    settings: Settings = Depends(get_settings),
):
    """OpenAI-compatible chat completions — raw passthrough."""
    body = await raw_request.body()
    parsed = _json_loads(body)

    if not parsed.get("model"):
        parsed["model"] = get_active_model()

    is_stream = parsed.get("stream", False)

    try:
        if is_stream:
            return StreamingResponse(
                _stream_raw(provider, parsed),
                media_type="text/event-stream",
                headers={
                    "X-Accel-Buffering": "no",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            return await _complete_raw(provider, parsed)

    except ProviderError:
        raise
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OpenAI passthrough error: {e}")
        raise HTTPException(
            status_code=getattr(e, "status_code", 500),
            detail={"error": {"message": str(e), "type": "api_error"}},
        )


async def _complete_raw(provider, body: dict):
    """Non-streaming: raw httpx POST, return response bytes directly."""
    nim = provider
    await nim._acquire_request_slot()
    try:
        await nim._global_rate_limiter.wait_if_blocked()

        model = body.get("model", get_active_model())
        candidates = nim._candidate_models(model)
        last_error = None
        start = time.time()

        for candidate in candidates:
            attempted: set[str] = set()
            while True:
                lease = await nim._key_manager.acquire(exclude=attempted)
                if lease is None:
                    break

                key = lease.key
                attempted.add(key)
                try:
                    req = dict(body)
                    req["model"] = candidate

                    resp = await nim._http_client.post(
                        f"{nim._base_url}/chat/completions",
                        content=_json_bytes(req),
                        headers={
                            "Authorization": f"Bearer {key}",
                            "Content-Type": "application/json",
                            "Accept": "application/json",
                        },
                    )

                    if resp.status_code == 200:
                        await nim._key_manager.record_success(key)
                        nim._remember_working_model(candidate)
                        _log_admin(candidate, key, "success", (time.time() - start) * 1000)
                        # Return raw bytes — no parsing
                        return Response(
                            content=resp.content,
                            media_type="application/json",
                            status_code=200,
                        )

                    # Handle errors
                    if resp.status_code == 429:
                        retry_after = resp.headers.get("retry-after")
                        cooldown = int(retry_after) if retry_after else None
                        await nim._key_manager.record_rate_limit(key, cooldown_seconds=cooldown)
                    if resp.status_code >= 500:
                        await nim._get_model_cb(candidate)._record_failure()
                    if resp.status_code == 404:
                        break  # next model
                    if resp.status_code in (400, 422):
                        raise HTTPException(
                            status_code=resp.status_code,
                            detail=_json_loads(resp.content),
                        )
                    last_error = Exception(f"HTTP {resp.status_code}: {resp.text[:200]}")
                    _log_admin(candidate, key, "failed", (time.time() - start) * 1000, str(last_error)[:200])
                    continue

                except HTTPException:
                    raise
                except Exception as e:
                    last_error = e
                    logger.warning("OPENAI_RAW: model=%s failed: %s", candidate, str(e)[:200])
                    continue
                finally:
                    await lease.release()

        if last_error:
            raise HTTPException(
                status_code=502,
                detail={"error": {"message": str(last_error), "type": "api_error"}},
            )
        wait = await nim._key_manager.min_wait_seconds()
        raise HTTPException(
            status_code=429,
            detail={"error": {"message": f"All keys busy. Retry in {wait:.1f}s.", "type": "rate_limit_error"}},
        )
    finally:
        await nim._release_request_slot()


async def _stream_raw(provider, body: dict):
    """Streaming: raw httpx SSE relay. Zero parsing — just bytes through."""
    nim = provider
    await nim._acquire_request_slot()
    try:
        await nim._global_rate_limiter.wait_if_blocked()

        model = body.get("model", get_active_model())
        candidates = nim._candidate_models(model)
        last_error = None
        succeeded = False

        for candidate in candidates:
            attempted: set[str] = set()
            while True:
                lease = await nim._key_manager.acquire(exclude=attempted)
                if lease is None:
                    break

                key = lease.key
                attempted.add(key)
                try:
                    req = dict(body)
                    req["model"] = candidate
                    req["stream"] = True

                    # Raw streaming — no SDK, no Pydantic, just bytes
                    async with nim._http_client.stream(
                        "POST",
                        f"{nim._base_url}/chat/completions",
                        content=_json_bytes(req),
                        headers={
                            "Authorization": f"Bearer {key}",
                            "Content-Type": "application/json",
                            "Accept": "text/event-stream",
                        },
                    ) as resp:
                        if resp.status_code != 200:
                            error_body = await resp.aread()
                            if resp.status_code == 429:
                                retry_after = resp.headers.get("retry-after")
                                cooldown = int(retry_after) if retry_after else None
                                await nim._key_manager.record_rate_limit(key, cooldown_seconds=cooldown)
                            if resp.status_code >= 500:
                                await nim._get_model_cb(candidate)._record_failure()
                            if resp.status_code == 404:
                                break  # next model
                            if resp.status_code in (400, 422):
                                yield f"data: {error_body.decode()}\n\n"
                                yield "data: [DONE]\n\n"
                                succeeded = True
                                break
                            last_error = Exception(f"HTTP {resp.status_code}")
                            continue

                        # Relay raw SSE lines directly — zero parsing
                        async for line in resp.aiter_lines():
                            if line:
                                yield f"{line}\n\n"

                    await nim._key_manager.record_success(key)
                    nim._remember_working_model(candidate)
                    succeeded = True
                    break

                except Exception as e:
                    last_error = e
                    sc = nim._status_code(e)
                    if sc == 429:
                        await nim._key_manager.record_rate_limit(key)
                    if nim._should_record_model_failure(e):
                        await nim._get_model_cb(candidate)._record_failure()
                    logger.warning("OPENAI_STREAM_RAW: model=%s failed: %s", candidate, str(e)[:200])
                    if nim._is_bad_request_error(e):
                        err = {"error": {"message": str(e), "type": "invalid_request_error"}}
                        yield f"data: {_json_bytes(err).decode()}\n\n"
                        yield "data: [DONE]\n\n"
                        succeeded = True
                        break
                    if nim._is_model_not_found_error(e):
                        break
                    continue
                finally:
                    await lease.release()

            if succeeded:
                break

        if not succeeded:
            msg = str(last_error) if last_error else "All API keys busy"
            err = {"error": {"message": msg, "type": "api_error"}}
            yield f"data: {_json_bytes(err).decode()}\n\n"
            yield "data: [DONE]\n\n"

    finally:
        await nim._release_request_slot()


# =============================================================================
# /v1/models
# =============================================================================


@router.get("/v1/models")
async def list_models():
    """OpenAI-compatible model listing."""
    active = get_active_model()
    fallbacks = get_configured_fallback_models()

    seen = set()
    all_models = []
    for m in [active] + fallbacks:
        if m not in seen:
            seen.add(m)
            all_models.append(m)

    return {
        "object": "list",
        "data": [
            {
                "id": model,
                "object": "model",
                "created": 0,
                "owned_by": model.split("/")[0] if "/" in model else "nvidia",
            }
            for model in all_models
        ],
    }


# =============================================================================
# /health/deep  — Item 12: verify upstream liveness
# =============================================================================


@router.get("/health/deep")
async def deep_health(
    provider: BaseProvider = Depends(get_provider),
):
    """Deep health check: verify at least one key can reach NIM."""
    nim = provider
    try:
        lease = await nim._key_manager.acquire()
        if lease is None:
            return {
                "status": "degraded",
                "detail": "No API keys available",
                "upstream": "unknown",
            }
        try:
            resp = await nim._http_client.post(
                f"{nim._base_url}/chat/completions",
                content=_json_bytes({
                    "model": get_active_model(),
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                    "stream": False,
                }),
                headers={
                    "Authorization": f"Bearer {lease.key}",
                    "Content-Type": "application/json",
                },
            )
            upstream_ok = resp.status_code == 200
            return {
                "status": "healthy" if upstream_ok else "degraded",
                "upstream": f"HTTP {resp.status_code}",
                "model": get_active_model(),
            }
        finally:
            await lease.release()
    except Exception as e:
        return {
            "status": "unhealthy",
            "detail": str(e)[:200],
            "upstream": "unreachable",
        }
