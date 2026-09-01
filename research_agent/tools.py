"""Typed function-tool definition and router for multi-source search."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any

from pydantic import Field, StrictInt, StrictStr

from .models import StrictModel
from .schema import strict_json_schema
from .security import sanitize_search_output

if TYPE_CHECKING:
    from .sources import MultiSourceSearchTool


class SearchSourcesArgs(StrictModel):
    """Arguments accepted by the model's search_sources function call."""

    query: StrictStr = Field(min_length=1, max_length=1000)
    max_results: StrictInt = Field(ge=1, le=50)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return {
            str(key): _jsonable(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return str(value)


SEARCH_SOURCES_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "search_sources",
    "description": (
        "Search the configured independent sources. Use the returned stable IDs "
        "when citing evidence in the final report."
    ),
    "parameters": strict_json_schema(SearchSourcesArgs),
    "strict": True,
}


def search_sources_tool_definition() -> dict[str, Any]:
    """Return a copy of the Responses API function-tool definition."""

    return json.loads(json.dumps(SEARCH_SOURCES_TOOL))


class SearchSourcesRouter:
    """Validate and dispatch model calls to ``MultiSourceSearchTool``."""

    name = "search_sources"

    def __init__(self, search_tool: MultiSourceSearchTool):
        self.search_tool = search_tool

    def dispatch(self, name: str, arguments: Mapping[str, Any] | str) -> Any:
        if name != self.name:
            raise ValueError(f"unknown tool: {name}")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ValueError("search_sources arguments are not valid JSON") from exc
        parsed = SearchSourcesArgs.model_validate(arguments)
        return sanitize_search_output(
            _jsonable(self.search_tool.search(parsed.query, limit=parsed.max_results))
        )

    def dispatch_call(self, call: Any) -> Any:
        """Dispatch a Responses SDK or dictionary function-call item."""

        name = _read(call, "name")
        arguments = _read(call, "arguments", {})
        return self.dispatch(name, arguments)


ToolRouter = SearchSourcesRouter


def _read(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def route_search_sources(
    search_tool: MultiSourceSearchTool,
    arguments: Mapping[str, Any] | str,
) -> Any:
    """Convenience function for dispatching one search_sources call."""

    return SearchSourcesRouter(search_tool).dispatch("search_sources", arguments)
