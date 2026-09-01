import json
import unittest
from types import SimpleNamespace

from research_agent.agent import AgentError, ResearchAgent
from research_agent.models import Source
from research_agent.retry import RetryPolicy
from research_agent.sources import FixtureProvider, MultiSourceSearchTool


class FakeResponsesModel:
    model = "fake-model"

    def __init__(self, citation_ids):
        self.citation_ids = citation_ids
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        if len(self.requests) == 1:
            return {
                "output": [
                    {
                        "type": "function_call",
                        "name": "search_sources",
                        "call_id": "call-1",
                        "arguments": json.dumps({"query": "test topic", "max_results": 4}),
                    }
                ]
            }
        return {
            "output_text": json.dumps(
                {
                    "executive_summary": "Two sources agree on the central point.",
                    "key_findings": [
                        {
                            "finding_id": "f1",
                            "statement": "The evidence supports the central point.",
                            "importance": "high",
                            "confidence": 0.9,
                            "citation_ids": self.citation_ids,
                        }
                    ],
                    "comparison": [
                        {
                            "dimension": "evidence scope",
                            "consensus": "Both sources support the central point.",
                            "disagreements": [],
                            "source_views": [
                                {"source_id": self.citation_ids[0], "position": "Supports it."},
                                {"source_id": self.citation_ids[1], "position": "Adds context."},
                            ],
                        }
                    ],
                    "limitations": ["Small fixture corpus."],
                }
            )
        }


def build_test_agent(model):
    sources = [
        Source(
            id="source-a",
            provider="academic",
            kind="academic",
            title="Academic source",
            url="https://example.com/a",
            snippet="A",
        ),
        Source(
            id="source-b",
            provider="practitioner",
            kind="engineering",
            title="Practitioner source",
            url="https://example.com/b",
            snippet="B",
        ),
    ]
    search = MultiSourceSearchTool(
        [
            FixtureProvider(fixtures=[sources[0]], source_id="academic"),
            FixtureProvider(fixtures=[sources[1]], source_id="practitioner"),
        ],
        retry_policy=RetryPolicy(max_attempts=1, jitter=0),
    )
    return ResearchAgent(
        model=model,
        search_tool=search,
        model_name="fake-model",
        max_turns=2,
        max_attempts=1,
        retry_initial_delay=0,
    )


class AgentTests(unittest.TestCase):
    def test_settings_retry_count_includes_initial_attempt(self):
        attempts = []
        sleeps = []

        class AlwaysTimeoutModel:
            def create(self, **request):
                attempts.append(request)
                raise TimeoutError("temporary provider failure")

        settings = SimpleNamespace(
            max_retries=2,
            max_tool_rounds=1,
            initial_retry_delay=0.25,
            max_retry_delay=0.5,
        )
        agent = ResearchAgent(
            model=AlwaysTimeoutModel(),
            search_tool=object(),
            settings=settings,
            sleep=sleeps.append,
            use_planner=False,
        )

        with self.assertRaises(TimeoutError):
            agent._model_call([])

        self.assertEqual(len(attempts), 3)
        self.assertEqual(sleeps, [0.25, 0.5])

    def test_executes_tool_loop_and_returns_grounded_structured_report(self):
        model = FakeResponsesModel(["source-a", "source-b"])
        report = build_test_agent(model).run("test topic")

        self.assertEqual(len(model.requests), 2)
        self.assertEqual(model.requests[0]["tool_choice"], "required")
        self.assertEqual(
            model.requests[0]["text"]["format"]["type"],
            "json_schema",
        )
        self.assertEqual(len(report.tool_calls), 1)
        self.assertGreaterEqual(len(report.tool_calls[0].planned_queries), 1)
        self.assertLessEqual(len(report.tool_calls[0].planned_queries), 4)
        self.assertEqual(report.tool_calls[0].stop_reason, "query_cap_reached")
        self.assertGreaterEqual(report.tool_calls[0].retrieval_coverage, 0.0)
        self.assertEqual(report.audit.grounding_score, 1.0)
        self.assertEqual(report.audit.provider_count, 2)
        self.assertIsNotNone(report.verification)
        self.assertTrue(report.verification.claim_checks)
        self.assertGreaterEqual(report.audit.score, 0.8)

    def test_rejects_unknown_citation_ids(self):
        model = FakeResponsesModel(["source-a", "not-returned"])
        with self.assertRaises(AgentError):
            build_test_agent(model).run("test topic")
