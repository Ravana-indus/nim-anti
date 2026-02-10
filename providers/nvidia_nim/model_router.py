"""Model routing and health tracking for NVIDIA NIM fallback chain."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List

DEFAULT_NIM_MODEL_CHAIN = [
    "kimi-coding/k2p5",
    "litellm-proxy/minimaxai/minimax-m2.1",
    "litellm-proxy/z-ai/glm4.7",
]


@dataclass
class ModelHealth:
    total: int = 0
    success: int = 0
    recent_errors: Deque[str] = field(default_factory=lambda: deque(maxlen=10))

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 1.0
        return self.success / self.total


class NimModelRouter:
    """Routes requests through a primary->fallback model chain."""

    def __init__(self, model_chain: Iterable[str] | None = None):
        chain = [m.strip() for m in (model_chain or DEFAULT_NIM_MODEL_CHAIN) if m and m.strip()]
        if not chain:
            raise ValueError("NVIDIA NIM model chain cannot be empty")
        self._chain: List[str] = chain
        self._health: Dict[str, ModelHealth] = {m: ModelHealth() for m in chain}

    @property
    def chain(self) -> list[str]:
        return list(self._chain)

    def get_model_chain(self) -> list[str]:
        """Return models in fallback order.

        Current strategy preserves explicit config order while collecting health metrics.
        """
        return list(self._chain)

    def record_success(self, model: str) -> None:
        h = self._health.setdefault(model, ModelHealth())
        h.total += 1
        h.success += 1

    def record_error(self, model: str, error_text: str) -> None:
        h = self._health.setdefault(model, ModelHealth())
        h.total += 1
        h.recent_errors.append(error_text)

    def get_health(self, model: str) -> ModelHealth:
        return self._health.setdefault(model, ModelHealth())
