"""Async retry queue for rate-limited requests.

Enqueues requests that fail due to rate limiting and retries them
after the key cooldown period, with configurable max retries and
exponential backoff.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, Optional

logger = logging.getLogger(__name__)


@dataclass
class RetryItem:
    """A request waiting to be retried."""
    attempt: int = 0
    max_retries: int = 2
    backoff_base: float = 1.5
    enqueued_at: float = field(default_factory=time.monotonic)
    next_retry_at: float = 0.0

    @property
    def delay_seconds(self) -> float:
        """Exponential backoff delay for the current attempt."""
        return self.backoff_base ** self.attempt

    def should_retry(self) -> bool:
        """Check if this item has remaining retry attempts."""
        return self.attempt < self.max_retries

    def schedule_next(self) -> None:
        """Calculate next retry time using exponential backoff."""
        self.attempt += 1
        self.next_retry_at = time.monotonic() + self.delay_seconds


class RetryQueue:
    """Async retry queue for rate-limited API requests.

    Usage:
        queue = RetryQueue(max_retries=2, backoff_base=1.5)

        # When a request is rate-limited:
        if queue.can_retry(item):
            await queue.enqueue(item, retry_fn)

        # The retry_fn will be called after the backoff period.
    """

    def __init__(
        self,
        max_retries: int = 2,
        backoff_base: float = 1.5,
        max_queue_size: int = 50,
    ):
        self.max_retries = max(0, max_retries)
        self.backoff_base = max(1.0, backoff_base)
        self.max_queue_size = max(1, max_queue_size)
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._active_tasks: set[asyncio.Task] = set()
        self._total_retried = 0
        self._total_succeeded = 0
        self._total_exhausted = 0

    async def enqueue(
        self,
        retry_fn: Callable[[], Awaitable[Any]],
        on_success: Optional[Callable[[Any], Awaitable[None]]] = None,
        on_failure: Optional[Callable[[Exception], Awaitable[None]]] = None,
        attempt: int = 0,
    ) -> bool:
        """Enqueue a retry with exponential backoff.

        Args:
            retry_fn: Async function to retry
            on_success: Callback on successful retry
            on_failure: Callback when all retries are exhausted
            attempt: Current attempt number

        Returns:
            True if enqueued, False if queue is full or max retries reached
        """
        if attempt >= self.max_retries:
            self._total_exhausted += 1
            logger.warning(f"RetryQueue: max retries ({self.max_retries}) exhausted")
            return False

        if self._queue.full():
            logger.warning(f"RetryQueue: queue full ({self.max_queue_size}), dropping request")
            return False

        item = RetryItem(
            attempt=attempt,
            max_retries=self.max_retries,
            backoff_base=self.backoff_base,
        )
        item.schedule_next()

        task = asyncio.create_task(
            self._process_retry(item, retry_fn, on_success, on_failure)
        )
        self._active_tasks.add(task)
        task.add_done_callback(self._active_tasks.discard)

        self._total_retried += 1
        logger.info(
            f"RetryQueue: scheduled retry #{item.attempt} "
            f"in {item.delay_seconds:.1f}s"
        )
        return True

    async def _process_retry(
        self,
        item: RetryItem,
        retry_fn: Callable[[], Awaitable[Any]],
        on_success: Optional[Callable[[Any], Awaitable[None]]],
        on_failure: Optional[Callable[[Exception], Awaitable[None]]],
    ) -> None:
        """Wait for backoff, then retry."""
        wait_time = max(0, item.next_retry_at - time.monotonic())
        if wait_time > 0:
            await asyncio.sleep(wait_time)

        try:
            result = await retry_fn()
            self._total_succeeded += 1
            logger.info(f"RetryQueue: retry #{item.attempt} succeeded")
            if on_success:
                await on_success(result)
        except Exception as e:
            logger.warning(f"RetryQueue: retry #{item.attempt} failed: {e}")
            if item.should_retry():
                # Re-enqueue with incremented attempt
                await self.enqueue(
                    retry_fn,
                    on_success=on_success,
                    on_failure=on_failure,
                    attempt=item.attempt,
                )
            else:
                self._total_exhausted += 1
                if on_failure:
                    await on_failure(e)

    async def cancel_all(self) -> int:
        """Cancel all pending retries."""
        cancelled = 0
        for task in list(self._active_tasks):
            if not task.done():
                task.cancel()
                cancelled += 1
        self._active_tasks.clear()
        return cancelled

    def snapshot(self) -> dict:
        """Return queue status for monitoring."""
        return {
            "pending": len(self._active_tasks),
            "max_queue_size": self.max_queue_size,
            "max_retries": self.max_retries,
            "backoff_base": self.backoff_base,
            "stats": {
                "total_retried": self._total_retried,
                "total_succeeded": self._total_succeeded,
                "total_exhausted": self._total_exhausted,
            },
        }
