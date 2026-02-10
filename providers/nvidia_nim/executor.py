"""Resilient request executor for NVIDIA NIM."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional

from .key_manager import NimKeyManager
from .model_router import NimModelRouter
from .resilience import AttemptError, AggregatedNimError, classify_error, with_backoff_retry


class NimResilientExecutor:
    """Tries model fallback chain across multiple API keys with retries."""

    def __init__(
        self,
        key_manager: NimKeyManager,
        model_router: NimModelRouter,
        call_fn: Callable[[str, str, dict[str, Any], bool], Awaitable[Any]],
        per_attempt_retries: int = 2,
    ):
        self._key_manager = key_manager
        self._model_router = model_router
        self._call_fn = call_fn
        self._per_attempt_retries = max(1, per_attempt_retries)

    async def execute(self, body: dict[str, Any], stream: bool = False) -> Any:
        attempts: list[AttemptError] = []

        for model in self._model_router.get_model_chain():
            attempted_keys: set[str] = set()

            while True:
                lease = await self._key_manager.acquire(exclude=attempted_keys)
                if lease is None:
                    break
                key = lease.key
                attempted_keys.add(key)

                request_body = dict(body)
                request_body["model"] = model

                try:
                    result = await with_backoff_retry(
                        lambda: self._call_fn(model, key, request_body, stream),
                        should_retry=lambda exc: classify_error(exc)[0],
                        attempts=self._per_attempt_retries,
                    )
                    self._model_router.record_success(model)
                    await self._key_manager.record_success(key)
                    return result
                except Exception as exc:  # noqa: BLE001
                    retryable, fatal = classify_error(exc)
                    self._model_router.record_error(model, str(exc))

                    if getattr(exc, "status_code", None) == 429:
                        await self._key_manager.record_rate_limit(key)

                    attempts.append(
                        AttemptError(
                            model=model,
                            key_suffix=key[-4:] if len(key) >= 4 else key,
                            error_type=type(exc).__name__,
                            message=str(exc),
                            retryable=retryable,
                        )
                    )

                    if fatal:
                        break
                finally:
                    await lease.release()

        raise AggregatedNimError(attempts)
