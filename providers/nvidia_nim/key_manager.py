"""API key management and per-key throttling for NVIDIA NIM."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, Optional, Set


@dataclass
class KeyLease:
    """Represents a leased key that must be released."""

    manager: "NimKeyManager"
    key: str
    released: bool = False

    async def release(self) -> None:
        if not self.released:
            await self.manager.release(self.key)
            self.released = True


class NimKeyManager:
    """Manages key selection with rate limits, cooldowns, and load balancing."""

    def __init__(
        self,
        api_keys: Iterable[str],
        rate_limit: int = 40,
        rate_window: int = 60,
        cooldown_seconds: int = 60,
    ):
        keys = [k.strip() for k in api_keys if k and k.strip()]
        if not keys:
            raise ValueError("At least one NVIDIA NIM API key is required")

        self._keys = keys
        self._rate_limit = max(1, rate_limit)
        self._rate_window = max(1, rate_window)
        self._cooldown_seconds = max(1, cooldown_seconds)

        self._requests: Dict[str, Deque[float]] = {k: deque() for k in keys}
        self._cooldown_until: Dict[str, float] = {k: 0.0 for k in keys}
        self._in_flight: Dict[str, int] = {k: 0 for k in keys}
        self._rr_index = 0
        self._lock = asyncio.Lock()

    @property
    def keys(self) -> list[str]:
        return list(self._keys)

    def get_available_keys(self) -> list[str]:
        """Get list of available keys (not in cooldown)."""
        now = time.monotonic()
        return [k for k in self._keys if self._is_available(k, now)]

    def _prune(self, key: str, now: float) -> None:
        window_start = now - self._rate_window
        q = self._requests[key]
        while q and q[0] <= window_start:
            q.popleft()

    def _is_available(self, key: str, now: float) -> bool:
        self._prune(key, now)
        if self._cooldown_until[key] > now:
            return False
        return len(self._requests[key]) < self._rate_limit

    async def acquire(self, exclude: Optional[Set[str]] = None) -> Optional[KeyLease]:
        """Acquire the best available key, or None if none are usable right now."""
        excluded = exclude or set()
        now = time.monotonic()

        async with self._lock:
            candidates = [k for k in self._keys if k not in excluded and self._is_available(k, now)]
            if not candidates:
                return None

            # Prefer highest remaining capacity and lowest in-flight.
            def score(k: str) -> tuple[int, int]:
                remaining = self._rate_limit - len(self._requests[k])
                return (remaining, -self._in_flight[k])

            best_remaining = max(score(k)[0] for k in candidates)
            tied = [k for k in candidates if score(k)[0] == best_remaining]
            min_in_flight = min(self._in_flight[k] for k in tied)
            tied = [k for k in tied if self._in_flight[k] == min_in_flight]

            # Round-robin fallback among equals.
            if len(tied) > 1:
                start = self._rr_index % len(self._keys)
                order = self._keys[start:] + self._keys[:start]
                for k in order:
                    if k in tied:
                        selected = k
                        break
                self._rr_index = (self._keys.index(selected) + 1) % len(self._keys)
            else:
                selected = tied[0]
                self._rr_index = (self._keys.index(selected) + 1) % len(self._keys)

            self._requests[selected].append(now)
            self._in_flight[selected] += 1
            return KeyLease(manager=self, key=selected)

    async def release(self, key: str) -> None:
        async with self._lock:
            if key in self._in_flight and self._in_flight[key] > 0:
                self._in_flight[key] -= 1

    async def record_rate_limit(self, key: str, cooldown_seconds: Optional[int] = None) -> None:
        async with self._lock:
            cooldown = cooldown_seconds or self._cooldown_seconds
            self._cooldown_until[key] = max(self._cooldown_until.get(key, 0), time.monotonic() + cooldown)

    async def record_success(self, key: str) -> None:
        async with self._lock:
            self._cooldown_until[key] = 0.0
