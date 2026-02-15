"""Circuit Breaker pattern implementation for fault tolerance.

Prevents cascading failures by temporarily blocking requests to a failing service.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable, Awaitable, Any

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"      # Failing, requests are blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitStats:
    """Statistics for circuit breaker monitoring."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    consecutive_failures: int = 0
    state_changes: int = 0


class CircuitBreaker:
    """
    Circuit breaker implementation with async support.
    
    States:
    - CLOSED: Normal operation. Requests pass through.
    - OPEN: Circuit is broken. Requests fail fast.
    - HALF_OPEN: Testing recovery. Limited requests allowed.
    
    Configuration:
    - failure_threshold: Number of consecutive failures to trip the circuit
    - recovery_timeout: Seconds to wait before attempting recovery
    - half_open_max_calls: Max requests allowed in half-open state
    - success_threshold: Consecutive successes needed to close circuit
    """
    
    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
        success_threshold: int = 2,
        on_state_change: Optional[Callable[[CircuitState, CircuitState], Awaitable[None]]] = None,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.success_threshold = success_threshold
        self.on_state_change = on_state_change
        
        self._state = CircuitState.CLOSED
        self._last_state_change = time.monotonic()
        self._stats = CircuitStats()
        self._half_open_calls = 0
        self._lock = asyncio.Lock()
        
        logger.info(
            f"CircuitBreaker '{name}' initialized: "
            f"failure_threshold={failure_threshold}, "
            f"recovery_timeout={recovery_timeout}s"
        )
    
    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state
    
    @property
    def stats(self) -> CircuitStats:
        """Circuit breaker statistics."""
        return self._stats
    
    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        return self._state == CircuitState.OPEN
    
    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self._state == CircuitState.HALF_OPEN
    
    def _should_allow_request(self) -> bool:
        """Check if request should be allowed based on current state."""
        if self._state == CircuitState.CLOSED:
            return True
        
        if self._state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            elapsed = time.monotonic() - self._last_state_change
            if elapsed >= self.recovery_timeout:
                return True  # Will transition to half-open
            return False
        
        if self._state == CircuitState.HALF_OPEN:
            # Allow limited requests in half-open state
            return self._half_open_calls < self.half_open_max_calls
        
        return False
    
    async def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state with logging and callback."""
        old_state = self._state
        if old_state == new_state:
            return
        
        self._state = new_state
        self._last_state_change = time.monotonic()
        self._stats.state_changes += 1
        
        logger.warning(
            f"CircuitBreaker '{self.name}' state change: "
            f"{old_state.value} -> {new_state.value}"
        )
        
        if self.on_state_change:
            try:
                await self.on_state_change(old_state, new_state)
            except Exception as e:
                logger.error(f"CircuitBreaker on_state_change callback error: {e}")
    
    async def _check_state_transition(self) -> None:
        """Check and perform state transitions based on current conditions."""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_state_change
            if elapsed >= self.recovery_timeout:
                await self._transition_to(CircuitState.HALF_OPEN)
                self._half_open_calls = 0
    
    async def call(self, fn: Callable[[], Awaitable[Any]]) -> Any:
        """
        Execute a function through the circuit breaker.
        
        Args:
            fn: Async function to execute
            
        Returns:
            Result of the function
            
        Raises:
            CircuitBreakerOpen: If circuit is open and not ready for recovery
            Exception: Any exception from the function call
        """
        async with self._lock:
            await self._check_state_transition()
            
            if not self._should_allow_request():
                raise CircuitBreakerOpen(
                    f"CircuitBreaker '{self.name}' is open. "
                    f"Retry after {self.recovery_timeout - (time.monotonic() - self._last_state_change):.1f}s"
                )
            
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1
        
        # Execute outside lock to allow concurrent calls
        self._stats.total_requests += 1
        
        try:
            result = await fn()
            await self._record_success()
            return result
        except Exception as e:
            await self._record_failure()
            raise
    
    async def _record_success(self) -> None:
        """Record a successful request."""
        async with self._lock:
            self._stats.successful_requests += 1
            self._stats.last_success_time = time.monotonic()
            self._stats.consecutive_failures = 0
            
            if self._state == CircuitState.HALF_OPEN:
                # Check if we have enough successes to close
                recent_successes = self._stats.successful_requests
                if recent_successes >= self.success_threshold:
                    await self._transition_to(CircuitState.CLOSED)
    
    async def _record_failure(self) -> None:
        """Record a failed request."""
        async with self._lock:
            self._stats.failed_requests += 1
            self._stats.last_failure_time = time.monotonic()
            self._stats.consecutive_failures += 1
            
            if self._state == CircuitState.HALF_OPEN:
                # Failure in half-open immediately opens circuit
                await self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                # Check if we should open circuit
                if self._stats.consecutive_failures >= self.failure_threshold:
                    await self._transition_to(CircuitState.OPEN)
    
    async def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        async with self._lock:
            await self._transition_to(CircuitState.CLOSED)
            self._stats.consecutive_failures = 0
            self._half_open_calls = 0
            logger.info(f"CircuitBreaker '{self.name}' manually reset")
    
    async def trip(self) -> None:
        """Manually trip the circuit breaker to open state."""
        async with self._lock:
            await self._transition_to(CircuitState.OPEN)
            logger.warning(f"CircuitBreaker '{self.name}' manually tripped")
    
    def snapshot(self) -> dict:
        """Get a snapshot of circuit breaker state for monitoring."""
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "stats": {
                "total_requests": self._stats.total_requests,
                "successful_requests": self._stats.successful_requests,
                "failed_requests": self._stats.failed_requests,
                "consecutive_failures": self._stats.consecutive_failures,
                "state_changes": self._stats.state_changes,
                "last_failure_time": self._stats.last_failure_time,
                "last_success_time": self._stats.last_success_time,
            },
            "is_closed": self.is_closed,
            "is_open": self.is_open,
            "is_half_open": self.is_half_open,
        }


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open and request is blocked."""
    pass


class CircuitBreakerRegistry:
    """
    Global registry for circuit breakers.
    
    Allows managing multiple circuit breakers by name and
    provides aggregate monitoring.
    """
    
    _instance: Optional["CircuitBreakerRegistry"] = None
    
    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()
    
    @classmethod
    def get_instance(cls) -> "CircuitBreakerRegistry":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None
    
    async def get_or_create(
        self,
        name: str,
        **kwargs,
    ) -> CircuitBreaker:
        """Get existing circuit breaker or create new one."""
        async with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name=name, **kwargs)
            return self._breakers[name]
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self._breakers.get(name)
    
    def get_all(self) -> dict[str, CircuitBreaker]:
        """Get all circuit breakers."""
        return dict(self._breakers)
    
    async def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            await breaker.reset()
    
    def snapshot_all(self) -> dict[str, dict]:
        """Get snapshots of all circuit breakers."""
        return {name: breaker.snapshot() for name, breaker in self._breakers.items()}


# Convenience function for creating circuit breakers
def create_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    **kwargs,
) -> CircuitBreaker:
    """Create a new circuit breaker instance."""
    return CircuitBreaker(
        name=name,
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        **kwargs,
    )
