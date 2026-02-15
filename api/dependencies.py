"""Dependency injection for FastAPI."""

from typing import Optional

from config.settings import Settings, get_settings as _get_settings, NVIDIA_NIM_BASE_URL
from providers.base import BaseProvider, ProviderConfig


# Global provider instance (singleton)
_provider: Optional[BaseProvider] = None


def get_settings() -> Settings:
    """Get application settings via dependency injection."""
    return _get_settings()


def get_provider() -> BaseProvider:
    """Get or create the provider instance based on settings.provider_type."""
    global _provider
    if _provider is None:
        settings = get_settings()

        if settings.provider_type == "nvidia_nim":
            from providers.nvidia_nim import NvidiaNimProvider

            # Support both single key (backward compat) and multiple keys
            raw_api_keys = getattr(settings, "nvidia_nim_api_keys", "") or ""
            if isinstance(raw_api_keys, str) and raw_api_keys.strip():
                api_keys = [k.strip() for k in raw_api_keys.split(",") if k.strip()]
            else:
                api_keys = [settings.nvidia_nim_api_key] if settings.nvidia_nim_api_key else []

            config = ProviderConfig(
                api_key=settings.nvidia_nim_api_key,  # Single key (deprecated, use api_keys)
                api_keys=api_keys,  # Multiple keys for rotation
                base_url=NVIDIA_NIM_BASE_URL,
                rate_limit=settings.nvidia_nim_rate_limit,
                rate_window=settings.nvidia_nim_rate_window,
                key_cooldown_sec=settings.nvidia_nim_key_cooldown_seconds,
                max_in_flight=settings.nvidia_nim_max_in_flight,
                request_timeout_sec=settings.nvidia_nim_request_timeout_seconds,
                openai_max_retries=settings.nvidia_nim_openai_max_retries,
                nim_settings=settings.nim,
                # Circuit breaker settings
                circuit_breaker_threshold=getattr(settings, 'circuit_breaker_threshold', 5),
                circuit_breaker_recovery=getattr(settings, 'circuit_breaker_recovery_timeout', 30.0),
                # HTTP connection pooling
                max_connections=getattr(settings, 'http_max_connections', 100),
                max_keepalive_connections=getattr(settings, 'http_max_keepalive_connections', 20),
            )
            _provider = NvidiaNimProvider(config)
        else:
            raise ValueError(
                f"Unknown provider_type: '{settings.provider_type}'. "
                f"Supported: 'nvidia_nim'"
            )
    return _provider


async def cleanup_provider():
    """Cleanup provider resources."""
    global _provider
    if _provider:
        if hasattr(_provider, "aclose"):
            await _provider.aclose()
        else:
            client = getattr(_provider, "_client", None)
            if client and hasattr(client, "aclose"):
                await client.aclose()
    _provider = None
