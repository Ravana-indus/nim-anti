"""NVIDIA NIM provider implementation with multi-key and model fallback."""

import asyncio
import json
import logging
import time
import uuid
from threading import Lock
from typing import Any, AsyncIterator, Optional

import httpx
from openai import AsyncOpenAI

from config.settings import get_model_fallback_chain
from providers.base import BaseProvider, ProviderConfig
from providers.exceptions import RateLimitError
from providers.rate_limit import GlobalRateLimiter
from providers.circuit_breaker import CircuitBreaker, CircuitBreakerOpen
from .errors import map_error
from .key_manager import NimKeyManager
from .request import build_request_body
from .response import convert_response
from .utils import (
    ContentType,
    HeuristicToolParser,
    SSEBuilder,
    ThinkTagParser,
    map_stop_reason,
)

logger = logging.getLogger(__name__)

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
        self._request_semaphore = asyncio.Semaphore(self._max_in_flight)
        self._active_requests = 0
        self._active_requests_lock = asyncio.Lock()
        self._sticky_model_lock = Lock()
        self._sticky_model: Optional[str] = None

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

        # Circuit breaker for fault tolerance
        self._circuit_breaker = CircuitBreaker(
            name="nvidia_nim",
            failure_threshold=getattr(config, 'circuit_breaker_threshold', 5),
            recovery_timeout=getattr(config, 'circuit_breaker_recovery', 30.0),
        )

        # Shared HTTP client with connection pooling for better performance
        self._http_client = httpx.AsyncClient(
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

        # Backward-compatible primary client attribute.
        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            max_retries=self._openai_max_retries,
            timeout=self._request_timeout_sec,
            http_client=self._http_client,
        )

        # Async client cache per key.
        self._client_cache: dict[str, AsyncOpenAI] = {self._api_key: self._client}

        async def create_client_for_key(key: str) -> AsyncOpenAI:
            return AsyncOpenAI(
                api_key=key,
                base_url=self._base_url,
                max_retries=self._openai_max_retries,
                timeout=self._request_timeout_sec,
                http_client=self._http_client,
            )

        self._client_factory = create_client_for_key

    async def _get_client(self, key: str) -> AsyncOpenAI:
        """Get or create an AsyncOpenAI client for a specific key."""
        if key not in self._client_cache:
            self._client_cache[key] = await self._client_factory(key)
        return self._client_cache[key]

    async def _acquire_request_slot(self) -> None:
        await self._request_semaphore.acquire()
        async with self._active_requests_lock:
            self._active_requests += 1

    async def _release_request_slot(self) -> None:
        async with self._active_requests_lock:
            self._active_requests = max(0, self._active_requests - 1)
        self._request_semaphore.release()

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

    def _candidate_models(self, request_model: str) -> list[str]:
        """Ordered model fallback list for the current request."""
        try:
            chain = get_model_fallback_chain(request_model)
        except Exception:
            return [request_model]
        with self._sticky_model_lock:
            sticky_model = self._sticky_model
        if sticky_model and sticky_model in chain:
            return [sticky_model] + [model for model in chain if model != sticky_model]
        return chain

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

    def set_sticky_model(self, model: Optional[str]) -> None:
        """Set sticky preferred model explicitly (e.g., from admin quick switch)."""
        with self._sticky_model_lock:
            self._sticky_model = model

    def clear_sticky_model(self) -> None:
        """Clear sticky preferred model."""
        with self._sticky_model_lock:
            self._sticky_model = None

    def get_circuit_breaker_status(self) -> dict:
        """Get circuit breaker status for monitoring."""
        return self._circuit_breaker.snapshot()

    async def reset_circuit_breaker(self) -> None:
        """Manually reset the circuit breaker."""
        await self._circuit_breaker.reset()

    async def stream_response(
        self, request: Any, input_tokens: int = 0
    ) -> AsyncIterator[str]:
        """Stream response in Anthropic SSE format."""
        # Check circuit breaker first
        if self._circuit_breaker.is_open:
            message_id = f"msg_{uuid.uuid4()}"
            sse = SSEBuilder(message_id, request.model, input_tokens)
            yield sse.message_start()
            for event in sse.emit_error(
                f"Service temporarily unavailable. Circuit breaker open. "
                f"Retry in {self._circuit_breaker.recovery_timeout:.0f}s."
            ):
                yield event
            for event in sse.close_all_blocks():
                yield event
            yield sse.message_delta("error", 0)
            yield sse.message_stop()
            yield sse.done()
            return

        await self._acquire_request_slot()
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
            for model_index, model in enumerate(candidate_models):
                model_start = time.time()
                skip_current_model = False
                attempted_keys: set[str] = set()
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
                        body = dict(base_body)
                        body["model"] = model
                        client = await self._get_client(key)

                        stream = await client.chat.completions.create(**body, stream=True)
                        async for chunk in stream:
                            if getattr(chunk, "usage", None):
                                usage_info = chunk.usage

                            if not chunk.choices:
                                continue

                            choice = chunk.choices[0]
                            delta = choice.delta

                            if choice.finish_reason:
                                finish_reason = choice.finish_reason

                            reasoning = getattr(delta, "reasoning_content", None)
                            if reasoning:
                                for event in sse.ensure_thinking_block():
                                    yield event
                                yield sse.emit_thinking_delta(reasoning)

                            if delta.content:
                                for part in think_parser.feed(delta.content):
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
                                                json.dumps(tool_use["input"]),
                                            )
                                            yield sse.content_block_stop(block_idx)

                            if delta.tool_calls:
                                for event in sse.close_content_blocks():
                                    yield event
                                for tc in delta.tool_calls:
                                    tc_info = {
                                        "index": tc.index,
                                        "id": tc.id,
                                        "function": {
                                            "name": tc.function.name,
                                            "arguments": tc.function.arguments,
                                        },
                                    }
                                    for event in self._process_tool_call(tc_info, sse):
                                        yield event

                        stream_succeeded = True
                        await self._key_manager.record_success(key)
                        # Record success for circuit breaker
                        await self._circuit_breaker._record_success()
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
                        if status_code == 429:
                            await self._key_manager.record_rate_limit(key)
                        # Record failure for circuit breaker (only for server errors)
                        if status_code and status_code >= 500:
                            await self._circuit_breaker._record_failure()
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
                            error=str(e)[:200],
                        )
                        logger.warning(
                            f"NIM_STREAM: model={model} key={key[:20]}... failed: {e}"
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
                    json.dumps(tool_use["input"]),
                )
                yield sse.content_block_stop(block_idx)

            if not error_occurred and sse.blocks.text_index == -1 and not sse.blocks.tool_indices:
                for event in sse.ensure_text_block():
                    yield event
                yield sse.emit_text_delta(" ")

            for event in sse.close_all_blocks():
                yield event

            output_tokens = (
                usage_info.completion_tokens
                if usage_info and hasattr(usage_info, "completion_tokens")
                else sse.estimate_output_tokens()
            )
            yield sse.message_delta(map_stop_reason(finish_reason), output_tokens)
            yield sse.message_stop()
            yield sse.done()
        finally:
            await self._release_request_slot()

    async def complete(self, request: Any) -> dict:
        """Make a non-streaming completion request with multi-key rotation."""
        await self._acquire_request_slot()
        try:
            await self._global_rate_limiter.wait_if_blocked()
            base_body = self._build_request_body(request, stream=False)

            initial_model = base_body.get("model", request.model)
            candidate_models = self._candidate_models(initial_model)
            last_error = None
            start_time = time.time()
            used_model = initial_model
            log_request = _get_admin_logger()
            abort_all = False
            for model_index, model in enumerate(candidate_models):
                model_start = time.time()
                skip_current_model = False
                attempted_keys: set[str] = set()
                while True:
                    lease = await self._key_manager.acquire(exclude=attempted_keys)
                    if lease is None:
                        break

                    key = lease.key
                    attempted_keys.add(key)
                    attempt_start = time.time()
                    try:
                        used_model = model
                        body = dict(base_body)
                        body["model"] = model
                        client = await self._get_client(key)
                        response = await client.chat.completions.create(**body)

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

                        return response.model_dump()
                    except Exception as e:
                        last_error = e
                        status_code = self._status_code(e)
                        if status_code == 429:
                            await self._key_manager.record_rate_limit(key)
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
                            error=str(e)[:200],
                        )
                        logger.warning(
                            f"NIM_COMPLETE: model={model} key={key[:20]}... failed: {e}"
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
        """Close all open OpenAI clients."""
        closed_ids: set[int] = set()
        primary_client = getattr(self, "_client", None)
        if primary_client is not None and hasattr(primary_client, "aclose"):
            await primary_client.aclose()
            closed_ids.add(id(primary_client))

        for client in self._client_cache.values():
            client_id = id(client)
            if client_id in closed_ids:
                continue
            if hasattr(client, "aclose"):
                await client.aclose()
            closed_ids.add(client_id)

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
                "client_pool_size": len(self._client_cache),
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
            tc_index = len(sse.blocks.tool_indices)

        fn_delta = tc.get("function", {})
        if fn_delta.get("name") is not None:
            sse.blocks.tool_names[tc_index] = (
                sse.blocks.tool_names.get(tc_index, "") + fn_delta["name"]
            )

        if tc_index not in sse.blocks.tool_indices:
            name = sse.blocks.tool_names.get(tc_index, "")
            if name or tc.get("id"):
                tool_id = tc.get("id") or f"tool_{uuid.uuid4()}"
                yield sse.start_tool_block(tc_index, tool_id, name)
                sse.blocks.tool_started[tc_index] = True
        elif not sse.blocks.tool_started.get(tc_index) and sse.blocks.tool_names.get(
            tc_index
        ):
            tool_id = tc.get("id") or f"tool_{uuid.uuid4()}"
            name = sse.blocks.tool_names[tc_index]
            yield sse.start_tool_block(tc_index, tool_id, name)
            sse.blocks.tool_started[tc_index] = True

        args = fn_delta.get("arguments", "")
        if args:
            if not sse.blocks.tool_started.get(tc_index):
                tool_id = tc.get("id") or f"tool_{uuid.uuid4()}"
                name = sse.blocks.tool_names.get(tc_index, "tool_call") or "tool_call"

                yield sse.start_tool_block(tc_index, tool_id, name)
                sse.blocks.tool_started[tc_index] = True

            if sse.blocks.tool_names.get(tc_index, "") == "Task":
                try:
                    args_json = json.loads(args)
                    if args_json.get("run_in_background") is not False:
                        logger.info(
                            "NIM_INTERCEPT: Forcing run_in_background=False for Task "
                            f"{tc.get('id', 'unknown')}"
                        )
                        args_json["run_in_background"] = False
                        args = json.dumps(args_json)
                except Exception as e:
                    logger.warning(
                        f"NIM_INTERCEPT: Failed to parse/modify Task args: {e}"
                    )

            yield sse.emit_tool_delta(tc_index, args)
