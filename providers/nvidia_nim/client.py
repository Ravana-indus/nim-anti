"""NVIDIA NIM provider implementation with multi-key and model fallback."""

import asyncio
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator, Optional

from openai import AsyncOpenAI

from providers.base import BaseProvider, ProviderConfig
from providers.exceptions import RateLimitError
from providers.rate_limit import GlobalRateLimiter
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
        self._nim_settings = config.nim_settings
        self._global_rate_limiter = GlobalRateLimiter.get_instance(
            rate_limit=config.rate_limit,
            rate_window=config.rate_window,
        )
        self._max_in_flight = max(1, config.max_in_flight)
        self._request_semaphore = asyncio.Semaphore(self._max_in_flight)
        self._active_requests = 0
        self._active_requests_lock = asyncio.Lock()

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

        # Backward-compatible primary client attribute.
        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            max_retries=2,
            timeout=300.0,
        )

        # Async client cache per key.
        self._client_cache: dict[str, AsyncOpenAI] = {self._api_key: self._client}

        async def create_client_for_key(key: str) -> AsyncOpenAI:
            return AsyncOpenAI(
                api_key=key,
                base_url=self._base_url,
                max_retries=2,
                timeout=300.0,
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

    async def stream_response(
        self, request: Any, input_tokens: int = 0
    ) -> AsyncIterator[str]:
        """Stream response in Anthropic SSE format."""
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
            model = base_body.get("model", request.model)
            stream_succeeded = False
            last_error = None
            start_time = time.time()
            used_key = None
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
                    if getattr(e, "status_code", None) == 429:
                        await self._key_manager.record_rate_limit(key)
                    await self._record_attempt(
                        model=model,
                        key=key,
                        status="failed",
                        latency_ms=(time.time() - attempt_start) * 1000,
                        stream=True,
                        error=e,
                    )
                    logger.warning(
                        f"NIM_STREAM: model={model} key={key[:20]}... failed: {e}"
                    )
                    continue
                finally:
                    await lease.release()

            if not stream_succeeded:
                response_time_ms = (time.time() - start_time) * 1000
                wait_seconds = await self._key_manager.min_wait_seconds()
                log_request = _get_admin_logger()
                log_request(
                    model=model,
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
                log_request = _get_admin_logger()
                log_request(
                    model=model,
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

            model = base_body.get("model", request.model)
            last_error = None
            start_time = time.time()

            attempted_keys: set[str] = set()
            while True:
                lease = await self._key_manager.acquire(exclude=attempted_keys)
                if lease is None:
                    break

                key = lease.key
                attempted_keys.add(key)
                attempt_start = time.time()
                try:
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
                    log_request = _get_admin_logger()
                    log_request(
                        model=model,
                        key=key,
                        status="success",
                        response_time_ms=response_time_ms,
                    )

                    return response.model_dump()
                except Exception as e:
                    last_error = e
                    if getattr(e, "status_code", None) == 429:
                        await self._key_manager.record_rate_limit(key)
                    await self._record_attempt(
                        model=model,
                        key=key,
                        status="failed",
                        latency_ms=(time.time() - attempt_start) * 1000,
                        stream=False,
                        error=e,
                    )

                    response_time_ms = (time.time() - start_time) * 1000
                    log_request = _get_admin_logger()
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
                    continue
                finally:
                    await lease.release()

            wait_seconds = await self._key_manager.min_wait_seconds()
            if last_error is None:
                raise RateLimitError(
                    f"All API keys are busy. Retry in about {wait_seconds:.1f}s."
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

        return {
            "keys": key_statuses,
            "health": {},
            "runtime": {
                "active_requests": active_requests,
                "max_in_flight": self._max_in_flight,
                "client_pool_size": len(self._client_cache),
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
