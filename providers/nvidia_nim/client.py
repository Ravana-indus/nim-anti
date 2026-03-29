"""NVIDIA NIM provider implementation with multi-key and model fallback."""

import asyncio
import heapq
try:
    import orjson
    def _fast_json_dumps(obj):
        return orjson.dumps(obj).decode()
    def _fast_json_loads(s):
        return orjson.loads(s)
except ImportError:
    import json
    def _fast_json_dumps(obj):
        return json.dumps(obj)
    def _fast_json_loads(s):
        return json.loads(s)
import logging
import time
import uuid
import hashlib
from collections import OrderedDict
from threading import Lock
from typing import Any, AsyncIterator, Optional

import httpx
import openai
from openai import AsyncOpenAI

from config.settings import get_model_fallback_chain
from providers.base import BaseProvider, ProviderConfig
from providers.exceptions import RateLimitError
from providers.rate_limit import GlobalRateLimiter
from providers.circuit_breaker import CircuitBreaker, CircuitBreakerOpen
from .errors import map_error
from .key_manager import NimKeyManager

logger = logging.getLogger(__name__)
from .request import build_request_body
from .response import convert_response
from .utils import (
    ContentType,
    HeuristicToolParser,
    SSEBuilder,
    ThinkTagParser,
    map_stop_reason,
)

# (logger moved near top)

# Lazy import for admin logging to avoid circular imports
_admin_log_request: Optional[callable] = None
_telemetry_attempt_recorder: Optional[callable] = None


def _get_admin_logger():
    """Get admin log_request function lazily to avoid circular imports."""
    global _admin_log_request
    if _admin_log_request is None:
        try:
            from api.admin import log_request as lr

            _admin_log_request = lr
        except ImportError:
            _admin_log_request = lambda *args, **kwargs: None
    return _admin_log_request


def _get_attempt_recorder():
    """Get telemetry attempt recorder lazily to avoid circular imports."""
    global _telemetry_attempt_recorder
    if _telemetry_attempt_recorder is None:
        try:
            from api.telemetry import telemetry

            _telemetry_attempt_recorder = telemetry.record_provider_attempt
        except ImportError:
            _telemetry_attempt_recorder = lambda *args, **kwargs: None
    return _telemetry_attempt_recorder


class PriorityRequestQueue:
    """Queue that prioritizing high-priority requests when concurrency limits are hit."""
    
    def __init__(self, max_concurrent: int = 10):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._priority_queue = []
        self._condition = asyncio.Condition()
        self._counter = 0  # Tie breaker
    
    async def acquire(self, priority: int = 0):
        """Higher priority values are processed first."""
        async with self._condition:
            self._counter += 1
            # heapq is a min-heap, so we negate priority
            heapq.heappush(self._priority_queue, (-priority, self._counter, asyncio.current_task()))
            while self._priority_queue[0][2] != asyncio.current_task() or self._semaphore.locked():
                if self._priority_queue[0][2] == asyncio.current_task() and not self._semaphore.locked():
                    break
                await self._condition.wait()
            
            await self._semaphore.acquire()
            heapq.heappop(self._priority_queue)
            self._condition.notify_all()

    def release(self):
        self._semaphore.release()
        async def _notify():
            async with self._condition:
                self._condition.notify_all()
        # Create a task to notify waiting coroutines
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_notify())
        except RuntimeError:
            pass


class NvidiaNimProvider(BaseProvider):
    """NVIDIA NIM provider with multi-key rotation fallback."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)

        self._api_key = ((config.api_keys[0] if config.api_keys else "") or config.api_key)
        self._base_url = (config.base_url or "https://integrate.api.nvidia.com/v1").rstrip(
            "/"
        )
        self._request_timeout_sec = max(10.0, float(getattr(config, "request_timeout_sec", 120.0)))
        self._openai_max_retries = max(0, int(getattr(config, "openai_max_retries", 0)))
        self._nim_settings = config.nim_settings
        self._global_rate_limiter = GlobalRateLimiter.get_instance(
            rate_limit=config.rate_limit,
            rate_window=config.rate_window,
        )
        self._max_in_flight = max(1, config.max_in_flight)
        self._request_queue = PriorityRequestQueue(max_concurrent=self._max_in_flight)
        self._active_requests = 0
        self._active_requests_lock = asyncio.Lock()
        self._sticky_model_lock = Lock()
        self._sticky_model: Optional[str] = None

        # Item 9: LRU Response Cache for non-streaming completions.
        # Key: (model, prompt_hash, temperature, top_p, tools_json)
        self._response_cache: OrderedDict[str, dict] = OrderedDict()
        self._max_cache_size = 100
        self._cache_lock = asyncio.Lock()

        # Initialize key manager with multiple API keys.
        if config.api_keys and len(config.api_keys) > 0:
            keys = config.api_keys
        else:
            keys = [config.api_key] if config.api_key else []

        self._key_manager = NimKeyManager(
            api_keys=keys,
            rate_limit=config.rate_limit or 40,
            rate_window=config.rate_window or 60,
            cooldown_seconds=config.key_cooldown_sec or 60,
        )

        # Per-model circuit breakers for fault tolerance.
        # Each model gets its own CB so one bad model doesn't block fallbacks.
        self._cb_failure_threshold = getattr(config, 'circuit_breaker_threshold', 5)
        self._cb_recovery_timeout = getattr(config, 'circuit_breaker_recovery', 30.0)
        self._model_circuit_breakers: dict[str, CircuitBreaker] = {}

        # Shared HTTP client with connection pooling and HTTP/2 for better performance
        self._http_client = httpx.AsyncClient(
            http2=True,  # Enable HTTP/2 multiplexing (requires h2 package)
            limits=httpx.Limits(
                max_connections=getattr(config, 'max_connections', 100),
                max_keepalive_connections=getattr(config, 'max_keepalive_connections', 20),
                keepalive_expiry=30.0,
            ),
            timeout=httpx.Timeout(
                connect=10.0,
                read=self._request_timeout_sec,
                write=30.0,
                pool=5.0,
            ),
        )

        # Backward-compatible attribute — kept for test assertions and admin,
        # but no longer used for actual API calls (raw httpx replaces SDK).
        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            max_retries=self._openai_max_retries,
            timeout=self._request_timeout_sec,
            http_client=self._http_client,
        )

    async def _acquire_request_slot(self, request: Any = None) -> None:
        # Simple heuristic: prioritize short messages
        priority = 10 if (request and hasattr(request, "messages") and len(request.messages) <= 2) else 0
        await self._request_queue.acquire(priority)
        async with self._active_requests_lock:
            self._active_requests += 1

    async def _release_request_slot(self) -> None:
        async with self._active_requests_lock:
            self._active_requests = max(0, self._active_requests - 1)
        self._request_queue.release()

    async def _record_attempt(
        self,
        model: str,
        key: str,
        status: str,
        latency_ms: float,
        stream: bool,
        error: Optional[Exception],
    ) -> None:
        recorder = _get_attempt_recorder()
        recorder(
            model=model,
            key_suffix=key[-4:] if len(key) >= 4 else key,
            status=status,
            latency_ms=latency_ms,
            stream=stream,
            error_type=(type(error).__name__ if error else None),
            error_message=(str(error)[:200] if error else None),
        )

    def _build_request_body(self, request: Any, stream: bool = False) -> dict:
        """Internal helper for tests and shared building."""
        return build_request_body(request, self._nim_settings, stream=stream)

    def _get_cache_key(self, model: str, body: dict) -> str:
        """Generate a stable cache key for a completion request."""
        # Focus on the core semantic inputs to the model.
        # We use orjson (fast) to serialize the values for hashing.
        messages = body.get("messages", [])
        tools = body.get("tools", [])
        temp = body.get("temperature", 0.0)
        top_p = body.get("top_p", 1.0)
        max_tokens = body.get("max_tokens", 4096)
        
        features = {
            "m": model,
            "msgs": messages,
            "t": tools,
            "temp": temp,
            "p": top_p,
            "max": max_tokens
        }
        feat_json = _fast_json_dumps(features)
        return hashlib.sha256(feat_json.encode()).hexdigest()

    @staticmethod
    def _flatten_body_for_raw(body: dict) -> bytes:
        """Flatten SDK-style body (with extra_body) into raw JSON bytes for httpx."""
        flat = {k: v for k, v in body.items() if k != "extra_body"}
        extra = body.get("extra_body")
        if extra:
            flat.update(extra)
        s = _fast_json_dumps(flat)
        return s.encode() if isinstance(s, str) else s

    def _get_model_cb(self, model: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a specific model."""
        if model not in self._model_circuit_breakers:
            self._model_circuit_breakers[model] = CircuitBreaker(
                name=f"model:{model}",
                failure_threshold=self._cb_failure_threshold,
                recovery_timeout=self._cb_recovery_timeout,
            )
        return self._model_circuit_breakers[model]

    def _candidate_models(self, request_model: str) -> list[str]:
        """Ordered model fallback list, skipping models with open circuit breakers."""
        try:
            chain = get_model_fallback_chain(request_model)
        except Exception:
            return [request_model]
        with self._sticky_model_lock:
            sticky_model = self._sticky_model
        if sticky_model and sticky_model in chain:
            ordered = [sticky_model] + [m for m in chain if m != sticky_model]
        else:
            ordered = chain

        # Filter out models whose circuit breaker is open
        available = [m for m in ordered if not self._get_model_cb(m).is_open]
        if not available:
            # All open — return the full list and let the CB recovery logic handle it
            logger.warning("All model circuit breakers are open — trying all models anyway")
            return ordered
        return available

    def _remember_working_model(self, model: str) -> None:
        with self._sticky_model_lock:
            self._sticky_model = model

    @staticmethod
    def _status_code(exc: Exception) -> Optional[int]:
        status = getattr(exc, "status_code", None)
        if isinstance(status, int):
            return status
        return None

    @classmethod
    def _is_bad_request_error(cls, exc: Exception) -> bool:
        status = cls._status_code(exc)
        if status in {400, 422}:
            return True
        message = str(exc).lower()
        return "invalid request" in message or "unprocessable" in message

    @classmethod
    def _is_model_not_found_error(cls, exc: Exception) -> bool:
        status = cls._status_code(exc)
        if status == 404:
            return True
        message = str(exc).lower()
        return "model not found" in message or "does not exist" in message

    @staticmethod
    def _is_timeout_error(exc: Exception) -> bool:
        return isinstance(exc, (openai.APITimeoutError, httpx.TimeoutException))

    @classmethod
    def _is_connection_error(cls, exc: Exception) -> bool:
        return isinstance(exc, (openai.APIConnectionError, httpx.TransportError)) and not cls._is_timeout_error(exc)

    @classmethod
    def _is_transport_error(cls, exc: Exception) -> bool:
        return cls._is_timeout_error(exc) or cls._is_connection_error(exc)

    @classmethod
    def _should_record_model_failure(cls, exc: Exception) -> bool:
        status = cls._status_code(exc)
        return bool((status and status >= 500) or cls._is_transport_error(exc))

    def _describe_attempt_error(self, exc: Exception) -> str:
        if self._is_timeout_error(exc):
            timeout_display = int(self._request_timeout_sec)
            return f"Timed out after {timeout_display}s"
        if self._is_connection_error(exc):
            return "Upstream connection error"
        return str(exc)

    def set_sticky_model(self, model: Optional[str]) -> None:
        """Set sticky preferred model explicitly (e.g., from admin quick switch)."""
        with self._sticky_model_lock:
            self._sticky_model = model

    def clear_sticky_model(self) -> None:
        """Clear sticky preferred model."""
        with self._sticky_model_lock:
            self._sticky_model = None

    def get_circuit_breaker_status(self) -> dict:
        """Get per-model circuit breaker status for monitoring."""
        return {
            model: cb.snapshot()
            for model, cb in self._model_circuit_breakers.items()
        }

    async def reset_circuit_breaker(self, model: Optional[str] = None) -> None:
        """Manually reset circuit breaker(s). If model is None, reset all."""
        if model:
            cb = self._model_circuit_breakers.get(model)
            if cb:
                await cb.reset()
        else:
            for cb in self._model_circuit_breakers.values():
                await cb.reset()

    async def stream_response(
        self, request: Any, input_tokens: int = 0
    ) -> AsyncIterator[str]:
        """Stream response in Anthropic SSE format."""
        # Per-model circuit breaker check is done inside _candidate_models()
        # which filters out models with open breakers.

        await self._acquire_request_slot(request)
        try:
            waited_reactively = await self._global_rate_limiter.wait_if_blocked()
            message_id = f"msg_{uuid.uuid4()}"
            sse = SSEBuilder(message_id, request.model, input_tokens)
            yield sse.message_start()
            if waited_reactively:
                for event in sse.emit_error("Global rate limit active. Resuming now..."):
                    yield event

            think_parser = ThinkTagParser()
            heuristic_parser = HeuristicToolParser()

            finish_reason = None
            usage_info = None
            error_occurred = False

            base_body = self._build_request_body(request, stream=True)
            initial_model = base_body.get("model", request.model)
            candidate_models = self._candidate_models(initial_model)
            stream_succeeded = False
            last_error = None
            start_time = time.time()
            used_key = None
            used_model = initial_model
            log_request = _get_admin_logger()
            abort_all = False

            # Pre-compute the base body bytes once (Item 6: avoid re-serializing on retries)
            base_body["stream"] = True
            # Item 11: Request usage info in streaming chunks for perfect accuracy
            base_body["stream_options"] = {"include_usage": True}
            _base_raw = self._flatten_body_for_raw(base_body)

            for model_index, model in enumerate(candidate_models):
                model_start = time.time()
                skip_current_model = False
                attempted_keys: set[str] = set()

                # Only re-serialize when model changes
                if model != initial_model:
                    body = dict(base_body)
                    body["model"] = model
                    body["stream"] = True
                    raw_body = self._flatten_body_for_raw(body)
                else:
                    raw_body = _base_raw

                while True:
                    lease = await self._key_manager.acquire(exclude=attempted_keys)
                    if lease is None:
                        break

                    key = lease.key
                    attempted_keys.add(key)
                    attempt_start = time.time()
                    try:
                        used_key = key
                        used_model = model

                        # Raw httpx streaming — no SDK Pydantic overhead
                        async with self._http_client.stream(
                            "POST",
                            f"{self._base_url}/chat/completions",
                            content=raw_body,
                            headers={
                                "Authorization": f"Bearer {key}",
                                "Content-Type": "application/json",
                                "Accept": "text/event-stream",
                            },
                        ) as raw_resp:
                            if raw_resp.status_code != 200:
                                err_body = await raw_resp.aread()
                                err_msg = err_body.decode(errors="replace")[:200]
                                exc = Exception(f"HTTP {raw_resp.status_code}: {err_msg}")
                                exc.status_code = raw_resp.status_code
                                # Preserve Retry-After for smarter 429 cooldowns
                                exc._retry_after = raw_resp.headers.get("retry-after")
                                raise exc

                            async for line in raw_resp.aiter_lines():
                                if not line.startswith("data: "):
                                    continue
                                payload = line[6:]
                                if payload.strip() == "[DONE]":
                                    break

                                chunk = _fast_json_loads(payload)

                                chunk_usage = chunk.get("usage")
                                if chunk_usage:
                                    usage_info = chunk_usage  # keep as dict — no dynamic class

                                choices = chunk.get("choices")
                                if not choices:
                                    continue

                                choice = choices[0]
                                delta = choice.get("delta") or {}

                                fr = choice.get("finish_reason")
                                if fr:
                                    finish_reason = fr

                                reasoning = delta.get("reasoning_content")
                                if reasoning:
                                    for event in sse.ensure_thinking_block():
                                        yield event
                                    yield sse.emit_thinking_delta(reasoning)
                                    # We don't 'continue' here anymore, because some models 
                                    # might send both fields in the same chunk (Item 14).

                                content = delta.get("content")
                                if content:
                                    for part in think_parser.feed(content):
                                        if part.type == ContentType.THINKING:
                                            for event in sse.ensure_thinking_block():
                                                yield event
                                            yield sse.emit_thinking_delta(part.content)
                                        else:
                                            filtered_text, detected_tools = heuristic_parser.feed(
                                                part.content
                                            )

                                            if filtered_text:
                                                for event in sse.ensure_text_block():
                                                    yield event
                                                yield sse.emit_text_delta(filtered_text)

                                            for tool_use in detected_tools:
                                                for event in sse.close_content_blocks():
                                                    yield event

                                                block_idx = sse.blocks.allocate_index()
                                                yield sse.content_block_start(
                                                    block_idx,
                                                    "tool_use",
                                                    id=tool_use["id"],
                                                    name=tool_use["name"],
                                                )
                                                yield sse.content_block_delta(
                                                    block_idx,
                                                    "input_json_delta",
                                                    _fast_json_dumps(tool_use["input"]),
                                                )
                                                yield sse.content_block_stop(block_idx)

                                tool_calls = delta.get("tool_calls")
                                if tool_calls:
                                    for event in sse.close_content_blocks():
                                        yield event
                                    for tc in tool_calls:
                                        fn = tc.get("function") or {}
                                        tc_info = {
                                            "index": tc.get("index", 0),
                                            "id": tc.get("id"),
                                            "function": {
                                                "name": fn.get("name"),
                                                "arguments": fn.get("arguments", ""),
                                            },
                                        }
                                        for event in self._process_tool_call(tc_info, sse):
                                            yield event
                                    
                                # Item 15: Ensure all blocks are closed and flushed on finish
                                if fr:
                                    rem = think_parser.flush()
                                    if rem:
                                        if rem.type == ContentType.THINKING:
                                            for event in sse.ensure_thinking_block():
                                                yield event
                                            yield sse.emit_thinking_delta(rem.content)
                                        else:
                                            for event in sse.ensure_text_block():
                                                yield event
                                            yield sse.emit_text_delta(rem.content)

                                    for event in sse.close_all_blocks():
                                        yield event

                        stream_succeeded = True
                        await self._key_manager.record_success(key)
                        # Record success for per-model circuit breaker
                        await self._get_model_cb(model)._record_success()
                        await self._record_attempt(
                            model=model,
                            key=key,
                            status="success",
                            latency_ms=(time.time() - attempt_start) * 1000,
                            stream=True,
                            error=None,
                        )
                        break

                    except Exception as e:
                        last_error = e
                        status_code = self._status_code(e)
                        error_text = self._describe_attempt_error(e)
                        if status_code == 429:
                            # Item 5: Parse Retry-After header if available
                            retry_after = getattr(e, '_retry_after', None)
                            cooldown = int(retry_after) if retry_after else None
                            await self._key_manager.record_rate_limit(key, cooldown_seconds=cooldown)
                        if self._should_record_model_failure(e):
                            await self._get_model_cb(model)._record_failure()
                        await self._record_attempt(
                            model=model,
                            key=key,
                            status="failed",
                            latency_ms=(time.time() - attempt_start) * 1000,
                            stream=True,
                            error=e,
                        )
                        log_request(
                            model=model,
                            key=key,
                            status="failed",
                            response_time_ms=(time.time() - attempt_start) * 1000,
                            error=error_text[:200],
                        )
                        logger.warning(
                            "NIM_STREAM: model=%s key=%s... failed: %s",
                            model,
                            key[:20],
                            error_text,
                        )
                        if self._is_bad_request_error(e):
                            abort_all = True
                            break
                        if self._is_model_not_found_error(e):
                            skip_current_model = True
                            break
                        continue
                    finally:
                        await lease.release()

                if stream_succeeded:
                    self._remember_working_model(model)
                    break
                if abort_all:
                    break
                if model_index < len(candidate_models) - 1:
                    log_request(
                        model=model,
                        key=used_key or "unknown",
                        status="fallback",
                        response_time_ms=(time.time() - model_start) * 1000,
                        error=f"Switching to {candidate_models[model_index + 1]}",
                    )
                if skip_current_model:
                    continue

            if not stream_succeeded:
                response_time_ms = (time.time() - start_time) * 1000
                wait_seconds = await self._key_manager.min_wait_seconds()
                log_request(
                    model=used_model,
                    key=used_key or "unknown",
                    status="failed",
                    response_time_ms=response_time_ms,
                    error=str(last_error)[:200]
                    if last_error
                    else f"No key available, retry in {wait_seconds:.1f}s",
                )

                for event in sse.close_content_blocks():
                    yield event
                error_text = (
                    str(map_error(last_error))
                    if last_error
                    else f"All API keys are busy. Retry in about {wait_seconds:.1f}s."
                )
                for event in sse.emit_error(error_text):
                    yield event
                error_occurred = True

            if stream_succeeded and used_key:
                response_time_ms = (time.time() - start_time) * 1000
                log_request(
                    model=used_model,
                    key=used_key,
                    status="success",
                    response_time_ms=response_time_ms,
                )

            remaining = think_parser.flush()
            if remaining:
                if remaining.type == ContentType.THINKING:
                    for event in sse.ensure_thinking_block():
                        yield event
                    yield sse.emit_thinking_delta(remaining.content)
                else:
                    for event in sse.ensure_text_block():
                        yield event
                    yield sse.emit_text_delta(remaining.content)

            for tool_use in heuristic_parser.flush():
                for event in sse.close_content_blocks():
                    yield event

                block_idx = sse.blocks.allocate_index()
                yield sse.content_block_start(
                    block_idx,
                    "tool_use",
                    id=tool_use["id"],
                    name=tool_use["name"],
                )
                yield sse.content_block_delta(
                    block_idx,
                    "input_json_delta",
                    _fast_json_dumps(tool_use["input"]),
                )
                yield sse.content_block_stop(block_idx)

            for event in self._flush_pending_tool_calls(sse):
                yield event

            if not error_occurred and sse.blocks.text_index == -1 and not sse.blocks.tool_indices:
                for event in sse.ensure_text_block():
                    yield event
                yield sse.emit_text_delta(" ")

            for event in sse.close_all_blocks():
                yield event

            output_tokens = (
                usage_info.get("completion_tokens")
                if isinstance(usage_info, dict)
                else None
            ) or sse.estimate_output_tokens()
            yield sse.message_delta(map_stop_reason(finish_reason), output_tokens)
            yield sse.message_stop()
            done_marker = sse.done()
            if done_marker:
                yield done_marker
        finally:
            await self._release_request_slot()

    async def complete(self, request: Any) -> dict:
        """Make a non-streaming completion request with multi-key rotation."""
        await self._acquire_request_slot(request)
        try:
            await self._global_rate_limiter.wait_if_blocked()
            base_body = self._build_request_body(request, stream=False)

            initial_model = base_body.get("model", request.model)
            
            # Item 9: Cache lookup
            cache_key = self._get_cache_key(initial_model, base_body)
            async with self._cache_lock:
                if cache_key in self._response_cache:
                    logger.debug("CACHE_HIT: %s", initial_model)
                    # Move to end (MRU)
                    cached_resp = self._response_cache.pop(cache_key)
                    self._response_cache[cache_key] = cached_resp
                    return cached_resp

            candidate_models = self._candidate_models(initial_model)
            last_error = None
            start_time = time.time()
            used_model = initial_model
            log_request = _get_admin_logger()
            abort_all = False

            # Pre-compute the base body bytes once
            _base_raw = self._flatten_body_for_raw(base_body)

            for model_index, model in enumerate(candidate_models):
                model_start = time.time()
                skip_current_model = False
                attempted_keys: set[str] = set()

                if model != initial_model:
                    body = dict(base_body)
                    body["model"] = model
                    raw_body = self._flatten_body_for_raw(body)
                else:
                    raw_body = _base_raw

                while True:
                    lease = await self._key_manager.acquire(exclude=attempted_keys)
                    if lease is None:
                        break

                    key = lease.key
                    attempted_keys.add(key)
                    attempt_start = time.time()
                    try:
                        used_model = model

                        # Raw httpx — no SDK Pydantic overhead
                        resp = await self._http_client.post(
                            f"{self._base_url}/chat/completions",
                            content=raw_body,
                            headers={
                                "Authorization": f"Bearer {key}",
                                "Content-Type": "application/json",
                                "Accept": "application/json",
                            },
                        )
                        if resp.status_code != 200:
                            err_msg = resp.text[:200]
                            exc = Exception(f"HTTP {resp.status_code}: {err_msg}")
                            exc.status_code = resp.status_code
                            exc._retry_after = resp.headers.get("retry-after")
                            raise exc

                        response_json = _fast_json_loads(resp.content)

                        await self._key_manager.record_success(key)
                        await self._record_attempt(
                            model=model,
                            key=key,
                            status="success",
                            latency_ms=(time.time() - attempt_start) * 1000,
                            stream=False,
                            error=None,
                        )

                        response_time_ms = (time.time() - start_time) * 1000
                        log_request(
                            model=model,
                            key=key,
                            status="success",
                            response_time_ms=response_time_ms,
                        )
                        self._remember_working_model(model)

                        # Item 9: Cache store
                        async with self._cache_lock:
                            if len(self._response_cache) >= self._max_cache_size:
                                self._response_cache.popitem(last=False)  # FIFO/LRU eviction
                            self._response_cache[cache_key] = response_json

                        return response_json
                    except Exception as e:
                        last_error = e
                        status_code = self._status_code(e)
                        error_text = self._describe_attempt_error(e)
                        if status_code == 429:
                            retry_after = getattr(e, '_retry_after', None)
                            cooldown = int(retry_after) if retry_after else None
                            await self._key_manager.record_rate_limit(key, cooldown_seconds=cooldown)
                        if self._should_record_model_failure(e):
                            await self._get_model_cb(model)._record_failure()
                        await self._record_attempt(
                            model=model,
                            key=key,
                            status="failed",
                            latency_ms=(time.time() - attempt_start) * 1000,
                            stream=False,
                            error=e,
                        )

                        # Log failed attempt latency, not cumulative request latency,
                        # to avoid inflating per-model averages in admin analytics.
                        response_time_ms = (time.time() - attempt_start) * 1000
                        log_request(
                            model=model,
                            key=key,
                            status="failed",
                            response_time_ms=response_time_ms,
                            error=error_text[:200],
                        )
                        logger.warning(
                            "NIM_COMPLETE: model=%s key=%s... failed: %s",
                            model,
                            key[:20],
                            error_text,
                        )
                        if self._is_bad_request_error(e):
                            abort_all = True
                            break
                        if self._is_model_not_found_error(e):
                            skip_current_model = True
                            break
                        continue
                    finally:
                        await lease.release()
                if abort_all:
                    break
                if model_index < len(candidate_models) - 1:
                    log_request(
                        model=model,
                        key="unknown",
                        status="fallback",
                        response_time_ms=(time.time() - model_start) * 1000,
                        error=f"Switching to {candidate_models[model_index + 1]}",
                    )
                if skip_current_model:
                    continue

            wait_seconds = await self._key_manager.min_wait_seconds()
            if last_error is None:
                raise RateLimitError(
                    f"All API keys are busy. Retry in about {wait_seconds:.1f}s."
                )
            logger.warning(
                "NIM_COMPLETE: all model fallbacks exhausted for model chain starting at %s; last model=%s",
                initial_model,
                used_model,
            )
            raise map_error(last_error)
        finally:
            await self._release_request_slot()

    async def aclose(self) -> None:
        """Close the shared HTTP client and legacy SDK client."""
        if hasattr(self, "_http_client") and self._http_client:
            await self._http_client.aclose()
        # Close legacy SDK client (kept for backward compat)
        client = getattr(self, "_client", None)
        if client is not None and hasattr(client, "aclose"):
            await client.aclose()

    async def get_admin_snapshot(self) -> dict:
        """Return provider runtime status for admin dashboards."""
        key_statuses = await self._key_manager.snapshot()
        async with self._active_requests_lock:
            active_requests = self._active_requests
        with self._sticky_model_lock:
            sticky_model = self._sticky_model

        return {
            "keys": key_statuses,
            "health": {},
            "runtime": {
                "active_requests": active_requests,
                "max_in_flight": self._max_in_flight,
                "sticky_model": sticky_model,
                "request_timeout_sec": self._request_timeout_sec,
                "openai_max_retries": self._openai_max_retries,
            },
        }

    def convert_response(self, response_json: dict, original_request: Any) -> Any:
        """Convert provider response to Anthropic format."""
        return convert_response(response_json, original_request)

    def _process_tool_call(self, tc: dict, sse: Any):
        """Process a single tool call delta and yield SSE events."""
        tc_index = tc.get("index", 0)
        if tc_index < 0:
            known_indices = (
                set(sse.blocks.tool_indices)
                | set(sse.blocks.tool_names)
                | set(sse.blocks.tool_contents)
                | set(sse.blocks.tool_ids)
            )
            tc_index = max(known_indices, default=-1) + 1

        fn_delta = tc.get("function", {})
        if fn_delta.get("name") is not None:
            sse.blocks.tool_names[tc_index] = (
                sse.blocks.tool_names.get(tc_index, "") + fn_delta["name"]
            )
        if tc.get("id"):
            sse.blocks.tool_ids[tc_index] = tc["id"]

        args = fn_delta.get("arguments", "")
        if sse.blocks.tool_started.get(tc_index):
            if args:
                yield sse.emit_tool_delta(tc_index, args)
            return

        if args:
            sse.blocks.tool_contents[tc_index] = (
                sse.blocks.tool_contents.get(tc_index, "") + args
            )

        name = sse.blocks.tool_names.get(tc_index, "")
        if not name:
            return

        if name == "Task":
            pending_args = sse.blocks.tool_contents.get(tc_index, "")
            if not pending_args:
                return
            try:
                args_json = _fast_json_loads(pending_args)
                if args_json.get("run_in_background") is not False:
                    logger.info(
                        "NIM_INTERCEPT: Forcing run_in_background=False for Task "
                        f"{tc.get('id', 'unknown')}"
                    )
                    args_json["run_in_background"] = False
                    pending_args = _fast_json_dumps(args_json)
                    sse.blocks.tool_contents[tc_index] = pending_args
            except Exception:
                return

            tool_id = sse.blocks.tool_ids.get(tc_index) or f"tool_{uuid.uuid4()}"
            yield sse.start_tool_block(tc_index, tool_id, name)
            block_idx = sse.blocks.tool_indices[tc_index]
            yield sse.content_block_delta(block_idx, "input_json_delta", pending_args)
            return

        if not args:
            return

        tool_id = sse.blocks.tool_ids.get(tc_index) or f"tool_{uuid.uuid4()}"
        buffered_args = sse.blocks.tool_contents.get(tc_index, "")
        yield sse.start_tool_block(tc_index, tool_id, name)
        block_idx = sse.blocks.tool_indices[tc_index]
        yield sse.content_block_delta(block_idx, "input_json_delta", buffered_args)

    def _flush_pending_tool_calls(self, sse: Any):
        """Flush buffered tool calls that never became startable mid-stream."""
        pending_indices = sorted(
            set(sse.blocks.tool_names)
            | set(sse.blocks.tool_contents)
            | set(sse.blocks.tool_ids)
        )
        for tc_index in pending_indices:
            if sse.blocks.tool_started.get(tc_index):
                continue

            name = sse.blocks.tool_names.get(tc_index, "")
            if not name:
                continue

            args = sse.blocks.tool_contents.get(tc_index, "")
            if name == "Task" and args:
                try:
                    args_json = _fast_json_loads(args)
                    if args_json.get("run_in_background") is not False:
                        logger.info(
                            "NIM_INTERCEPT: Forcing run_in_background=False for Task "
                            f"{sse.blocks.tool_ids.get(tc_index, 'unknown')}"
                        )
                        args_json["run_in_background"] = False
                        args = _fast_json_dumps(args_json)
                        sse.blocks.tool_contents[tc_index] = args
                except Exception as e:
                    logger.warning(
                        f"NIM_INTERCEPT: Failed to parse/modify Task args: {e}"
                    )

            if not args:
                args = "{}"
                sse.blocks.tool_contents[tc_index] = args

            tool_id = sse.blocks.tool_ids.get(tc_index) or f"tool_{uuid.uuid4()}"
            yield sse.start_tool_block(tc_index, tool_id, name)
            block_idx = sse.blocks.tool_indices[tc_index]
            yield sse.content_block_delta(block_idx, "input_json_delta", args)
