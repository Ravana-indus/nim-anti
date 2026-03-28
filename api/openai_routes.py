"""OpenAI-compatible passthrough endpoints.

Exposes /v1/chat/completions and /v1/models so tools expecting
an OpenAI-compatible API (Aider, Continue, Cursor, etc.) can use
cc-nim as a drop-in proxy. All requests flow through the same
key rotation, rate limiting, and circuit breaker infrastructure.
"""

import json
import logging
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from .dependencies import get_provider, get_settings
from config.settings import Settings, get_active_model, get_configured_fallback_models
from providers.base import BaseProvider
from providers.exceptions import ProviderError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["openai"])


# =============================================================================
# /v1/chat/completions  — OpenAI-compatible passthrough
# =============================================================================


@router.post("/v1/chat/completions")
async def chat_completions(
    raw_request: Request,
    provider: BaseProvider = Depends(get_provider),
    settings: Settings = Depends(get_settings),
):
    """OpenAI-compatible chat completions endpoint.

    Accepts standard OpenAI request format and proxies directly to
    NVIDIA NIM with key rotation and fallback.
    """
    body = await raw_request.json()

    # Default model to active model if not specified
    if not body.get("model"):
        body["model"] = get_active_model()

    is_stream = body.get("stream", False)

    try:
        if is_stream:
            return StreamingResponse(
                _stream_openai(provider, body),
                media_type="text/event-stream",
                headers={
                    "X-Accel-Buffering": "no",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            return await _complete_openai(provider, body)

    except ProviderError:
        raise
    except Exception as e:
        logger.error(f"OpenAI passthrough error: {e}")
        raise HTTPException(
            status_code=getattr(e, "status_code", 500),
            detail={"error": {"message": str(e), "type": "api_error"}},
        )


async def _complete_openai(provider, body: dict) -> dict:
    """Non-streaming OpenAI completion via provider key rotation."""
    nim_provider = provider  # NvidiaNimProvider

    await nim_provider._acquire_request_slot()
    try:
        await nim_provider._global_rate_limiter.wait_if_blocked()

        model = body.get("model", get_active_model())
        candidate_models = nim_provider._candidate_models(model)
        last_error = None
        start_time = time.time()

        for model_index, candidate in enumerate(candidate_models):
            attempted_keys: set[str] = set()
            while True:
                lease = await nim_provider._key_manager.acquire(exclude=attempted_keys)
                if lease is None:
                    break

                key = lease.key
                attempted_keys.add(key)
                try:
                    client = await nim_provider._get_client(key)
                    req_body = dict(body)
                    req_body["model"] = candidate

                    response = await client.chat.completions.create(**req_body)
                    await nim_provider._key_manager.record_success(key)
                    nim_provider._remember_working_model(candidate)
                    return response.model_dump()

                except Exception as e:
                    last_error = e
                    status_code = nim_provider._status_code(e)
                    if status_code == 429:
                        await nim_provider._key_manager.record_rate_limit(key)
                    if nim_provider._should_record_model_failure(e):
                        await nim_provider._get_model_cb(candidate)._record_failure()
                    logger.warning(
                        "OPENAI_COMPLETE: model=%s key=%s... failed: %s",
                        candidate, key[:20], str(e)[:200],
                    )
                    if nim_provider._is_bad_request_error(e):
                        raise HTTPException(status_code=400, detail={"error": {"message": str(e), "type": "invalid_request_error"}})
                    if nim_provider._is_model_not_found_error(e):
                        break  # skip to next model
                    continue
                finally:
                    await lease.release()

        if last_error:
            raise HTTPException(
                status_code=getattr(last_error, "status_code", 502),
                detail={"error": {"message": str(last_error), "type": "api_error"}},
            )
        wait_seconds = await nim_provider._key_manager.min_wait_seconds()
        raise HTTPException(
            status_code=429,
            detail={"error": {"message": f"All API keys busy. Retry in {wait_seconds:.1f}s.", "type": "rate_limit_error"}},
        )
    finally:
        await nim_provider._release_request_slot()


async def _stream_openai(provider, body: dict):
    """Streaming OpenAI completion — yields SSE chunks in OpenAI format."""
    nim_provider = provider

    await nim_provider._acquire_request_slot()
    try:
        await nim_provider._global_rate_limiter.wait_if_blocked()

        model = body.get("model", get_active_model())
        candidate_models = nim_provider._candidate_models(model)
        last_error = None
        stream_succeeded = False

        for model_index, candidate in enumerate(candidate_models):
            attempted_keys: set[str] = set()
            while True:
                lease = await nim_provider._key_manager.acquire(exclude=attempted_keys)
                if lease is None:
                    break

                key = lease.key
                attempted_keys.add(key)
                try:
                    client = await nim_provider._get_client(key)
                    req_body = dict(body)
                    req_body["model"] = candidate

                    stream = await client.chat.completions.create(**req_body, stream=True)
                    async for chunk in stream:
                        data = chunk.model_dump_json()
                        yield f"data: {data}\n\n"

                    yield "data: [DONE]\n\n"
                    await nim_provider._key_manager.record_success(key)
                    nim_provider._remember_working_model(candidate)
                    stream_succeeded = True
                    break

                except Exception as e:
                    last_error = e
                    status_code = nim_provider._status_code(e)
                    if status_code == 429:
                        await nim_provider._key_manager.record_rate_limit(key)
                    if nim_provider._should_record_model_failure(e):
                        await nim_provider._get_model_cb(candidate)._record_failure()
                    logger.warning(
                        "OPENAI_STREAM: model=%s key=%s... failed: %s",
                        candidate, key[:20], str(e)[:200],
                    )
                    if nim_provider._is_bad_request_error(e):
                        error_resp = {"error": {"message": str(e), "type": "invalid_request_error"}}
                        yield f"data: {json.dumps(error_resp)}\n\n"
                        yield "data: [DONE]\n\n"
                        stream_succeeded = True  # don't retry
                        break
                    if nim_provider._is_model_not_found_error(e):
                        break
                    continue
                finally:
                    await lease.release()

            if stream_succeeded:
                break

        if not stream_succeeded:
            error_msg = str(last_error) if last_error else "All API keys busy"
            error_resp = {"error": {"message": error_msg, "type": "api_error"}}
            yield f"data: {json.dumps(error_resp)}\n\n"
            yield "data: [DONE]\n\n"

    finally:
        await nim_provider._release_request_slot()


# =============================================================================
# /v1/models  — OpenAI-compatible model listing
# =============================================================================


@router.get("/v1/models")
async def list_models():
    """OpenAI-compatible model listing.

    Returns the active model + configured fallback models.
    """
    active = get_active_model()
    fallbacks = get_configured_fallback_models()

    # Dedupe while preserving order
    seen = set()
    all_models = []
    for m in [active] + fallbacks:
        if m not in seen:
            seen.add(m)
            all_models.append(m)

    models = [
        {
            "id": model,
            "object": "model",
            "created": 0,
            "owned_by": model.split("/")[0] if "/" in model else "nvidia",
        }
        for model in all_models
    ]

    return {"object": "list", "data": models}
