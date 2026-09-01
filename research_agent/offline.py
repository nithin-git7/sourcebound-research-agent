from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .agent import ResearchAgent
from .config import Settings
from .models import Source
from .retry import RetryPolicy
from .sources import FixtureProvider, MultiSourceSearchTool


class DeterministicResearchModel:
    """A tiny model double that exercises the real tool loop for local demos."""

    model = "offline-fixture-model"

    def create(self, **request: Any) -> dict[str, Any]:
        input_items = request.get("input", [])
        tool_outputs = [
            item for item in input_items if item.get("type") == "function_call_output"
        ]
        if not tool_outputs:
            question = next(
                (
                    str(item.get("content", ""))
                    for item in input_items
                    if item.get("role") == "user"
                ),
                "retrieval-augmented generation",
            )
            return {
                "output": [
                    {
                        "type": "function_call",
                        "name": "search_sources",
                        "call_id": "offline-search-1",
                        "arguments": json.dumps(
                            {"query": question, "max_results": 6}
                        ),
                    }
                ]
            }

        payload = json.loads(tool_outputs[-1]["output"])
        sources = payload.get("sources", [])
        source_ids = [source["id"] for source in sources]
        if not source_ids:
            raise RuntimeError("Fixture search returned no sources.")
        first = source_ids[0]
        second = source_ids[1] if len(source_ids) > 1 else first
        third = source_ids[2] if len(source_ids) > 2 else second
        draft = {
            "executive_summary": (
                "Across the academic, engineering, and survey-style sources, retrieval-"
                "augmented generation is presented as a way to ground model outputs in "
                "an updatable knowledge base. The trade-off is a larger runtime system: "
                "retrieval quality, latency, and corpus maintenance become part of answer quality."
            ),
            "key_findings": [
                {
                    "finding_id": "f1",
                    "statement": (
                        "RAG can add external evidence at answer time, which helps a model "
                        "work with knowledge that is newer or more domain-specific than its weights."
                    ),
                    "importance": "high",
                    "confidence": 0.9,
                    "citation_ids": [first, second],
                },
                {
                    "finding_id": "f2",
                    "statement": (
                        "RAG shifts failure modes toward retrieval and corpus quality: a weak "
                        "ranker or stale index can still produce a confident but poorly supported answer."
                    ),
                    "importance": "high",
                    "confidence": 0.86,
                    "citation_ids": [second, third],
                },
            ],
            "comparison": [
                {
                    "dimension": "Freshness versus system complexity",
                    "consensus": (
                        "The sources broadly agree that RAG improves updateability without "
                        "retraining the base model, while adding an operational retrieval layer."
                    ),
                    "disagreements": [
                        "The academic source emphasizes task-level grounding gains, while the "
                        "engineering source emphasizes latency and evaluation burden."
                    ],
                    "source_views": [
                        {
                            "source_id": first,
                            "position": "Frames retrieval as a way to supply task-relevant external knowledge.",
                        },
                        {
                            "source_id": second,
                            "position": "Highlights retrieval quality, latency, and maintenance as production risks.",
                        },
                        {
                            "source_id": third,
                            "position": "Places RAG on a spectrum between static model knowledge and external evidence.",
                        },
                    ],
                }
            ],
            "limitations": [
                "Offline mode uses a small curated fixture corpus rather than a live web search.",
                "The fixture demonstrates provenance and comparison behavior; it is not a current literature review.",
            ],
        }
        output_text = json.dumps(draft, ensure_ascii=False)
        return {
            "output_text": output_text,
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": output_text}],
                }
            ],
        }


def _fixture_sources() -> list[Source]:
    path = Path(__file__).with_name("data") / "research_sources.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    return [Source.model_validate(record) for record in records]


def build_offline_agent() -> ResearchAgent:
    records = _fixture_sources()
    grouped: dict[str, list[Source]] = {}
    for source in records:
        grouped.setdefault(source.provider, []).append(source)
    providers = [
        FixtureProvider(fixtures=provider_sources, source_id=provider_name)
        for provider_name, provider_sources in grouped.items()
    ]
    search_tool = MultiSourceSearchTool(
        providers,
        retry_policy=RetryPolicy(max_attempts=1, initial_delay=0, max_delay=0, jitter=0),
    )
    return ResearchAgent(
        model=DeterministicResearchModel(),
        search_tool=search_tool,
        settings=Settings(
            openai_api_key=None,
            openai_model="offline-fixture-model",
            openai_web_enabled=False,
        ),
        model_name="offline-fixture-model",
        max_turns=2,
        max_attempts=1,
        retry_initial_delay=0,
    )
