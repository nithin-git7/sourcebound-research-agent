"""Strict JSON Schema helpers for OpenAI Responses structured outputs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, TypeVar

from pydantic import BaseModel

from .models import ResearchReportDraft

ModelT = TypeVar("ModelT", bound=BaseModel)


def _make_strict(node: Any) -> Any:
    """Normalize a Pydantic schema to the Responses strict-schema subset."""

    if isinstance(node, list):
        return [_make_strict(item) for item in node]
    if not isinstance(node, dict):
        return node

    normalized = {key: _make_strict(value) for key, value in node.items()}
    normalized.pop("default", None)
    if normalized.get("type") == "object" or "properties" in normalized:
        normalized["additionalProperties"] = False
        properties = normalized.get("properties", {})
        normalized["required"] = list(properties)
    return normalized


def strict_json_schema(model: type[ModelT]) -> dict[str, Any]:
    """Build a closed JSON Schema with every object property required."""

    return _make_strict(deepcopy(model.model_json_schema(mode="validation")))


def report_draft_schema() -> dict[str, Any]:
    """Return the strict schema for :class:`ResearchReportDraft`."""

    return strict_json_schema(ResearchReportDraft)


def report_response_format(name: str = "research_report_draft") -> dict[str, Any]:
    """Return the value accepted as ``text.format`` by the Responses API."""

    return {
        "type": "json_schema",
        "name": name,
        "strict": True,
        "schema": report_draft_schema(),
    }


REPORT_DRAFT_SCHEMA = report_draft_schema()
REPORT_RESPONSE_FORMAT = report_response_format()
