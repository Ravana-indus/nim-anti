"""Model name normalization utilities.

Centralizes model name mapping logic to avoid duplication across the codebase.
"""

import os
import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

# Provider prefixes to strip from model names
_PROVIDER_PREFIXES = ["anthropic/", "openai/", "gemini/"]

# Claude model identifiers
_CLAUDE_IDENTIFIERS = ["haiku", "sonnet", "opus", "claude"]


@lru_cache(maxsize=1)
def _load_nim_model_catalog() -> list[str]:
    """Load local NVIDIA model IDs from nvidia_nim_models.json."""
    catalog_path = Path("nvidia_nim_models.json")
    if not catalog_path.exists():
        return []

    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        rows = payload.get("data", []) if isinstance(payload, dict) else []
        models: list[str] = []
        for row in rows:
            if isinstance(row, dict):
                model_id = row.get("id")
                if isinstance(model_id, str) and model_id.strip():
                    models.append(model_id.strip())
        return sorted(set(models))
    except Exception:
        return []


def resolve_model_alias(model: str) -> str:
    """Resolve shorthand model aliases to full NIM IDs when possible."""
    candidate = model.strip()
    if not candidate:
        return model

    catalog = _load_nim_model_catalog()
    if not catalog:
        return model

    # Exact match already valid.
    if candidate in catalog:
        return candidate

    # Providerless shorthand: match unique tail `<provider>/<name>`.
    tail_matches = [item for item in catalog if item.endswith(f"/{candidate}")]
    if len(tail_matches) == 1:
        return tail_matches[0]

    return model


def strip_provider_prefixes(model: str) -> str:
    """
    Strip provider prefixes from model name.

    Args:
        model: The model name, possibly with prefix

    Returns:
        Model name without provider prefix
    """
    for prefix in _PROVIDER_PREFIXES:
        if model.startswith(prefix):
            return model[len(prefix) :]
    return model


def is_claude_model(model: str) -> bool:
    """
    Check if a model name identifies as a Claude model.

    Args:
        model: The (prefix-stripped) model name

    Returns:
        True if this is a Claude model
    """
    model_lower = model.lower()
    return any(name in model_lower for name in _CLAUDE_IDENTIFIERS)


def normalize_model_name(model: str, default_model: Optional[str] = None) -> str:
    """
    Normalize a model name by stripping prefixes and mapping to default if needed.

    This is the central function for model name normalization across the API.
    It strips provider prefixes and maps Claude model names to the configured model.

    Args:
        model: The model name (may include provider prefix)
        default_model: The default model to use for Claude models.
                       If None, uses settings.model from config.

    Returns:
        Normalized model name (original if not a Claude model, mapped if Claude)
    """
    # Strip provider prefixes
    clean = strip_provider_prefixes(model)

    # Map Claude models to default
    if is_claude_model(clean):
        if default_model is None:
            # Use environment/config default
            default_model = os.getenv("MODEL", "moonshotai/kimi-k2-thinking")
        return default_model

    return resolve_model_alias(model)


def get_original_model(model: str) -> str:
    """
    Get the original model name, storing it before normalization.

    Convenience function that returns the input unchanged, intended to be
    called alongside normalize_model_name to capture the original.

    Args:
        model: The model name

    Returns:
        The model name unchanged (for documentation purposes)
    """
    return model
