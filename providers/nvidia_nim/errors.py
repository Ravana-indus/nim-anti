"""Error mapping for NVIDIA NIM provider."""

import openai
import httpx

from providers.exceptions import (
    AuthenticationError,
    InvalidRequestError,
    RateLimitError,
    OverloadedError,
    RequestTimeoutError,
    APIError,
)
from providers.rate_limit import GlobalRateLimiter


def _timeout_message(e: Exception) -> str:
    message = str(e).strip() or "Request timed out."
    if message.lower() == "request timed out.":
        return "Upstream request timed out."
    return message


def map_error(e: Exception) -> Exception:
    """Map OpenAI exception to specific ProviderError."""
    if isinstance(e, (openai.APITimeoutError, httpx.TimeoutException)):
        return RequestTimeoutError(_timeout_message(e), raw_error=str(e))
    if isinstance(e, openai.AuthenticationError):
        return AuthenticationError(str(e), raw_error=str(e))
    if isinstance(e, openai.RateLimitError):
        # Trigger global rate limit block
        GlobalRateLimiter.get_instance().set_blocked(60)  # Default 60s cooldown
        return RateLimitError(str(e), raw_error=str(e))
    if isinstance(e, openai.BadRequestError):
        return InvalidRequestError(str(e), raw_error=str(e))
    if isinstance(e, openai.APIConnectionError):
        if "timed out" in str(e).lower():
            return RequestTimeoutError(_timeout_message(e), raw_error=str(e))
        return APIError(str(e), status_code=503, raw_error=str(e))
    if isinstance(e, openai.InternalServerError):
        message = str(e)
        if "overloaded" in message.lower() or "capacity" in message.lower():
            return OverloadedError(message, raw_error=str(e))
        return APIError(message, status_code=500, raw_error=str(e))
    if isinstance(e, openai.APIError):
        return APIError(
            str(e), status_code=getattr(e, "status_code", 500), raw_error=str(e)
        )
    if isinstance(e, httpx.TransportError):
        if isinstance(e, httpx.TimeoutException) or "timed out" in str(e).lower():
            return RequestTimeoutError(_timeout_message(e), raw_error=str(e))
        return APIError(str(e), status_code=503, raw_error=str(e))

    return e
