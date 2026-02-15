"""Base provider interface - extend this to implement your own provider."""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Optional

from pydantic import BaseModel, Field

from config.nim import NimSettings


class ProviderConfig(BaseModel):
    """Configuration for a provider.

    Base fields apply to all providers. Provider-specific parameters
    (e.g. NIM temperature, top_p) are passed via nim_settings.
    """

    api_key: str
    api_keys: list[str] = Field(default_factory=list)
    base_url: Optional[str] = None
    rate_limit: Optional[int] = None
    rate_window: int = 60
    key_cooldown_sec: int = 60
    max_in_flight: int = 32
    request_timeout_sec: float = 120.0
    openai_max_retries: int = 0
    nim_settings: NimSettings = Field(default_factory=NimSettings)
    
    # Circuit breaker settings
    circuit_breaker_threshold: int = 5
    circuit_breaker_recovery: float = 30.0
    
    # HTTP connection pooling
    max_connections: int = 100
    max_keepalive_connections: int = 20


class BaseProvider(ABC):
    """Base class for all providers. Extend this to add your own."""

    def __init__(self, config: ProviderConfig):
        self.config = config

    @abstractmethod
    async def complete(self, request: Any) -> dict:
        """Make a non-streaming completion request. Returns raw JSON response."""
        pass

    @abstractmethod
    async def stream_response(
        self, request: Any, input_tokens: int = 0
    ) -> AsyncIterator[str]:
        """Stream response in Anthropic SSE format."""
        if False:
            yield ""

    @abstractmethod
    def convert_response(self, response_json: dict, original_request: Any) -> Any:
        """Convert provider response to Anthropic format."""
        pass
