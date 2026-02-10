"""Lazy AsyncOpenAI client pool keyed by API key."""

from __future__ import annotations

from typing import Dict

from openai import AsyncOpenAI


class NimClientPool:
    def __init__(self, base_url: str, timeout: float = 300.0, max_retries: int = 2):
        self._base_url = base_url
        self._timeout = timeout
        self._max_retries = max_retries
        self._clients: Dict[str, AsyncOpenAI] = {}

    def get_client(self, api_key: str) -> AsyncOpenAI:
        if api_key not in self._clients:
            self._clients[api_key] = AsyncOpenAI(
                api_key=api_key,
                base_url=self._base_url,
                max_retries=self._max_retries,
                timeout=self._timeout,
            )
        return self._clients[api_key]

    async def aclose(self) -> None:
        for client in self._clients.values():
            if hasattr(client, "aclose"):
                await client.aclose()
