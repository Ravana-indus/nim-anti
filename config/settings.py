"""Centralized configuration using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Optional

from pydantic import field_validator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

from .nim import NimSettings

load_dotenv()

# Fixed base URL for NVIDIA NIM
NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_NIM_MODEL_FALLBACK_ORDER = [
    "z-ai/glm5",
    "moonshotai/kimi-k2.5",
    "minimaxai/minimax-m2.1",
    "stepfun-ai/step-3.5-flash",
    "qwen/qwen3-next-80b-a3b-instruct",
    "qwen/qwen3-coder-480b-a35b-instruct",
    "qwen/qwen3.5-397b-a17b",
    "sarvamai/sarvam-m",
    "deepseek-ai/deepseek-v3_2",
]


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ==================== Provider Selection ====================
    provider_type: str = "nvidia_nim"

    # ==================== Messaging Platform Selection ====================
    messaging_platform: str = "telegram"

    # ==================== NVIDIA NIM Config ====================
    # Single key (backward compatibility) or comma-separated multiple keys
    nvidia_nim_api_key: str = ""
    nvidia_nim_api_keys: str = ""

    # ==================== Model Selection ====================
    # Backward-compatible single target model for Claude mapping.
    model: str = "moonshotai/kimi-k2.5"
    # Ordered model fallback chain. Comma-separated in env var.
    nvidia_nim_fallback_models: str = ",".join(DEFAULT_NIM_MODEL_FALLBACK_ORDER)
    # ==================== Rate Limiting (per-key) ====================
    nvidia_nim_rate_limit: int = 40
    nvidia_nim_rate_window: int = 60
    nvidia_nim_key_cooldown_seconds: int = 60
    nvidia_nim_max_in_flight: int = 32
    # Upstream request controls to prevent multi-minute retry cascades.
    nvidia_nim_request_timeout_seconds: float = 120.0
    nvidia_nim_openai_max_retries: int = 0

    # ==================== Fast Prefix Detection ====================
    fast_prefix_detection: bool = True

    # ==================== Optimizations ====================
    enable_network_probe_mock: bool = True
    enable_title_generation_skip: bool = True
    enable_suggestion_mode_skip: bool = True
    enable_filepath_extraction_mock: bool = True

    # ==================== NIM Settings ====================
    nim: NimSettings = Field(default_factory=NimSettings)

    # ==================== Bot Wrapper Config ====================
    telegram_bot_token: Optional[str] = None
    allowed_telegram_user_id: Optional[str] = None
    claude_workspace: str = "./agent_workspace"
    allowed_dir: str = ""
    max_cli_sessions: int = 10

    # ==================== Server ====================
    host: str = "0.0.0.0"
    port: int = 8085
    log_file: str = "server.log"
    request_timeout_seconds: float = 300.0  # 5 minutes default for request timeout middleware

    # ==================== Circuit Breaker ====================
    circuit_breaker_threshold: int = 5
    circuit_breaker_recovery_timeout: float = 30.0

    # ==================== HTTP Connection Pooling ====================
    http_max_connections: int = 100
    http_max_keepalive_connections: int = 20

    # Handle empty strings for optional string fields
    @field_validator(
        "telegram_bot_token",
        "allowed_telegram_user_id",
        mode="before",
    )
    @classmethod
    def parse_optional_str(cls, v):
        if v == "":
            return None
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


_active_model_lock = Lock()
_active_model_override: Optional[str] = None


def get_active_model() -> str:
    """Get currently active model (runtime override or configured default)."""
    from providers.model_utils import resolve_model_alias

    with _active_model_lock:
        if _active_model_override:
            return resolve_model_alias(_active_model_override)
    return resolve_model_alias(get_settings().model)


def set_active_model(model: str) -> str:
    """Set active model override at runtime."""
    from providers.model_utils import resolve_model_alias

    normalized = resolve_model_alias(model.strip())
    if not normalized:
        raise ValueError("model cannot be empty")
    with _active_model_lock:
        global _active_model_override
        _active_model_override = normalized
    return normalized


def clear_active_model_override() -> None:
    """Clear runtime override so configured MODEL is used."""
    with _active_model_lock:
        global _active_model_override
        _active_model_override = None


def has_active_model_override() -> bool:
    """Return True when runtime model override is set."""
    with _active_model_lock:
        return bool(_active_model_override)


def persist_model_to_env(model: str, env_path: str = ".env") -> str:
    """Persist MODEL to .env and refresh cached settings."""
    from providers.model_utils import resolve_model_alias

    normalized = resolve_model_alias(model.strip())
    if not normalized:
        raise ValueError("model cannot be empty")

    path = Path(env_path)
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
    else:
        lines = []

    replaced = False
    updated: list[str] = []
    for line in lines:
        if line.startswith("MODEL="):
            updated.append(f'MODEL="{normalized}"')
            replaced = True
        else:
            updated.append(line)

    if not replaced:
        if updated and updated[-1].strip():
            updated.append("")
        updated.append(f'MODEL="{normalized}"')

    path.write_text("\n".join(updated) + "\n", encoding="utf-8")
    get_settings.cache_clear()
    return normalized


def get_configured_fallback_models() -> list[str]:
    """Get configured fallback model order with canonical IDs."""
    from providers.model_utils import resolve_model_alias

    raw = get_settings().nvidia_nim_fallback_models
    models = [resolve_model_alias(part.strip()) for part in raw.split(",") if part.strip()]

    deduped: list[str] = []
    seen: set[str] = set()
    for model in models:
        if model not in seen:
            deduped.append(model)
            seen.add(model)
    return deduped


def get_model_fallback_chain(primary_model: Optional[str] = None) -> list[str]:
    """Get effective model attempt order with active/request model first."""
    from providers.model_utils import resolve_model_alias

    primary = resolve_model_alias((primary_model or get_active_model()).strip())
    configured = get_configured_fallback_models()
    return [primary] + [model for model in configured if model != primary]
