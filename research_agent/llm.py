"""Lazy OpenAI Responses SDK adapter and protocol for fakeable model clients."""

from __future__ import annotations

from typing import Any, Protocol


class ResponsesClient(Protocol):
    """The small interface the agent needs from a model client."""

    def create(self, **request: Any) -> Any:
        ...


class OpenAIResponsesAdapter:
    """Adapt ``OpenAI().responses.create`` without importing OpenAI eagerly."""

    def __init__(self, client: Any | None = None, *, api_key: str | None = None):
        self._client = client
        self._api_key = api_key

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise RuntimeError(
                    "The OpenAI SDK is required to use OpenAIResponsesAdapter"
                ) from exc
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def create(self, **request: Any) -> Any:
        """Create one response using the current OpenAI Responses API."""

        return self.client.responses.create(**request)

    complete = create


OpenAIResponsesLLM = OpenAIResponsesAdapter
