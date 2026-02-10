"""Resilience helpers for retries and aggregated errors."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List


@dataclass
class AttemptError:
    model: str
    key_suffix: str
    error_type: str
    message: str
    retryable: bool


class AggregatedNimError(Exception):
    """Raised when all model/key attempts fail."""

    def __init__(self, attempts: List[AttemptError]):
        self.attempts = attempts
        summary = " | ".join(
            [
                f"model={a.model} key=*{a.key_suffix} type={a.error_type} retryable={a.retryable} msg={a.message}"
                for a in attempts
            ]
        )
        super().__init__(f"All NVIDIA NIM attempts failed: {summary}")


def is_retryable_status(status_code: int | None) -> bool:
    if status_code is None:
        return False
    return status_code == 429 or 500 <= status_code < 600


def is_fatal_status(status_code: int | None) -> bool:
    return status_code in {401, 404}


def classify_error(exc: Exception) -> tuple[bool, bool]:
    status = getattr(exc, "status_code", None)
    message = str(exc).lower()

    retryable = is_retryable_status(status)
    fatal = is_fatal_status(status) or "model not found" in message

    if fatal:
        retryable = False
    return retryable, fatal


async def with_backoff_retry(
    fn: Callable[[], Awaitable[Any]],
    should_retry: Callable[[Exception], bool],
    attempts: int = 2,
    base_delay: float = 0.5,
) -> Any:
    """Retry a coroutine with linear backoff for retryable exceptions."""
    last_err: Exception | None = None
    for i in range(max(1, attempts)):
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if i == attempts - 1 or not should_retry(exc):
                raise
            await asyncio.sleep(base_delay * (i + 1))

    if last_err:
        raise last_err
