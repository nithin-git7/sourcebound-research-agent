import unittest

from pydantic import ValidationError

from research_agent.models import Source
from research_agent.planning import (
    FreshnessHint,
    RetrievalPlanner,
    StopCriteria,
    build_research_plan,
    decompose_question,
    rank_sources,
)


def source(
    source_id: str,
    provider: str,
    title: str,
    snippet: str,
    *,
    kind: str = "web",
    credibility: str = "unknown",
    published_at: str | None = "2026-08-01",
    url: str | None = None,
) -> Source:
    return Source(
        id=source_id,
        provider=provider,
        title=title,
        url=url or f"https://example.test/{source_id}",
        snippet=snippet,
        kind=kind,
        credibility=credibility,
        published_at=published_at,
    )


class PlanningTests(unittest.TestCase):
    def test_decomposition_is_small_focused_and_freshness_aware(self):
        intents = decompose_question(
            "What are the trade-offs of retrieval augmented generation?",
            max_query_count=3,
            freshness_hint="recent",
        )

        self.assertEqual(len(intents), 3)
        self.assertEqual(intents[0].intent_id, "overview")
        self.assertEqual(len({intent.query for intent in intents}), 3)
        self.assertTrue(all("recent" in intent.query for intent in intents))
        self.assertGreater(intents[0].priority, intents[-1].priority)

    def test_plan_contract_exposes_bounds_requirements_and_stop_criteria(self):
        plan = build_research_plan(
            "How do vector databases support semantic search?",
            required_source_kinds=["academic", "official", "academic"],
            freshness_hint=FreshnessHint(
                mode="current",
                max_age_days=180,
                reference_date="2026-09-01",
            ),
            max_query_count=2,
            per_query_result_limit=3,
            overall_budget=5,
            stop=StopCriteria(target_coverage=0.75, min_unique_sources=2, min_provider_count=2),
        )

        self.assertEqual(plan.required_source_kinds, ["academic", "official"])
        self.assertEqual(plan.max_query_count, 2)
        self.assertEqual(plan.per_query_result_limit, 3)
        self.assertEqual(plan.overall_budget, 5)
        self.assertEqual(plan.stop.min_provider_count, 2)
        with self.assertRaises(ValidationError):
            type(plan).model_validate({**plan.model_dump(), "unexpected": True})

    def test_ranking_deduplicates_and_prefers_relevance_credibility_and_freshness(self):
        plan = build_research_plan(
            "retrieval augmented generation quality",
            max_query_count=1,
            per_query_result_limit=6,
            overall_budget=6,
            freshness_hint=FreshnessHint(mode="recent", max_age_days=365, reference_date="2026-09-01"),
        )
        query = plan.intents[0].query
        results = {
            query: [
                source("weak", "blog", "Unrelated cooking guide", "Recipes and kitchen advice."),
                source(
                    "strong",
                    "academic",
                    "Retrieval augmented generation quality study",
                    "Evidence evaluates retrieval quality and grounded generation.",
                    kind="academic",
                    credibility="high",
                ),
                source(
                    "duplicate",
                    "blog",
                    "Retrieval augmented generation quality study",
                    "Same evidence mirrored elsewhere.",
                    url="https://example.test/strong/",
                ),
            ]
        }

        result = rank_sources(plan, results)

        self.assertEqual(result.raw_result_count, 3)
        self.assertEqual(result.deduplicated_count, 1)
        self.assertEqual(result.sources[0].source.id, "strong")
        self.assertEqual(result.sources[0].credibility_score, 1.0)
        self.assertGreater(result.sources[0].relevance_score, result.sources[-1].relevance_score)

    def test_ranking_injects_provider_diversity_when_scores_are_close(self):
        plan = build_research_plan(
            "climate adaptation policy",
            max_query_count=1,
            per_query_result_limit=4,
            overall_budget=4,
            stop={"min_unique_sources": 1, "min_provider_count": 2},
        )
        query = plan.intents[0].query
        results = {
            query: [
                source("a1", "provider-a", "Climate adaptation policy overview", "Climate adaptation policy evidence.", credibility="high"),
                source("a2", "provider-a-2", "Climate adaptation policy overview", "Climate adaptation policy evidence.", credibility="high"),
                source("b1", "provider-b", "Climate adaptation policy overview", "Climate adaptation policy evidence.", credibility="high"),
            ]
        }

        result = rank_sources(plan, results)

        self.assertEqual(result.provider_count, 3)
        self.assertEqual(result.sources[0].diversity_score, 1.0)
        self.assertEqual(result.sources[1].diversity_score, 1.0)
        self.assertEqual(result.sources[2].diversity_score, 1.0)

    def test_per_query_and_overall_caps_are_enforced(self):
        plan = build_research_plan(
            "distributed systems reliability",
            max_query_count=2,
            per_query_result_limit=2,
            overall_budget=3,
        )
        results = {
            plan.intents[0].query: [
                source("one", "p1", "Distributed systems reliability", "Reliability evidence."),
                source("two", "p1", "Distributed systems reliability", "Reliability evidence."),
                source("three", "p1", "Distributed systems reliability", "Reliability evidence."),
            ],
            plan.intents[1].query: [
                source("four", "p2", "Distributed systems reliability", "Reliability evidence."),
                source("five", "p2", "Distributed systems reliability", "Reliability evidence."),
            ],
            "an unplanned third query": [
                source("six", "p3", "Distributed systems reliability", "Reliability evidence."),
            ],
        }

        result = rank_sources(plan, results)

        self.assertEqual(result.raw_result_count, 3)
        self.assertEqual(len(result.sources), 3)
        self.assertEqual(result.budget_used, 3)
        self.assertLessEqual(result.queries_considered, 2)

    def test_stop_criteria_report_coverage_and_query_cap(self):
        plan = build_research_plan(
            "machine learning evaluation",
            max_query_count=1,
            per_query_result_limit=3,
            overall_budget=10,
            required_source_kinds=["academic", "official"],
            stop={"target_coverage": 0.8, "min_unique_sources": 2, "min_provider_count": 2},
        )
        query = plan.intents[0].query
        result = rank_sources(
            plan,
            {
                query: [
                    source("academic-1", "papers", "Machine learning evaluation study", "Machine learning evaluation methods.", kind="academic", credibility="high"),
                    source("academic-2", "papers", "Machine learning evaluation results", "Evaluation results and benchmarks.", kind="academic", credibility="high"),
                ]
            },
        )

        self.assertEqual(result.stop.reason, "query_cap_reached")
        self.assertTrue(result.stop.stop)
        self.assertIn("official", result.missing_source_kinds)
        self.assertLess(result.stop.coverage, 0.8)

    def test_needs_more_evidence_is_not_reported_as_complete(self):
        plan = build_research_plan(
            "privacy preserving machine learning",
            max_query_count=3,
            overall_budget=20,
            stop={"target_coverage": 0.95, "min_unique_sources": 4, "min_provider_count": 2},
        )
        result = rank_sources(
            plan,
            [source("only", "one-provider", "Privacy preserving machine learning", "Privacy preserving machine learning overview.")],
        )

        self.assertFalse(result.stop.stop)
        self.assertEqual(result.stop.reason, "needs_more_evidence")
        self.assertTrue(result.missing_intents)


if __name__ == "__main__":
    unittest.main()
