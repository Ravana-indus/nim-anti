"""Centralized configuration using Pydantic Settings."""

from functools import lru_cache
from typing import Optional

from pydantic import field_validator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

from .nim import NimSettings

load_dotenv()

# Fixed base URL for NVIDIA NIM
NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"


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
    # Primary -> fallback chain (comma-separated).
    nvidia_nim_model_chain: str = (
        "kimi-coding/k2p5,litellm-proxy/minimaxai/minimax-m2.1,litellm-proxy/z-ai/glm4.7"
    )

    # ==================== Rate Limiting (per-key) ====================
    nvidia_nim_rate_limit: int = 40
    nvidia_nim_rate_window: int = 60
    nvidia_nim_key_cooldown_seconds: int = 60

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
