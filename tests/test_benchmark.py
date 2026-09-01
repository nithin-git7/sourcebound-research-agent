import json
import unittest
from pathlib import Path

from research_agent.evaluation import (
    DEFAULT_BENCHMARK,
    DEFAULT_CASES,
    FIXTURE_METRIC_DISCLOSURE,
    evaluate_report,
    load_benchmark,
    run_evaluation_suite,
)
from research_agent.offline import build_offline_agent


class BenchmarkTests(unittest.TestCase):
    def test_curated_benchmark_has_metadata_concepts_and_criteria(self):
        benchmark = load_benchmark()

        self.assertEqual(benchmark.mode, "fixture")
        self.assertEqual(benchmark.benchmark_id, DEFAULT_BENCHMARK.benchmark_id)
        self.assertIn("deterministic proxy metrics", benchmark.metric_disclosure)
        self.assertEqual(len(benchmark.cases), 3)
        self.assertTrue(all(case.expected_concepts for case in benchmark.cases))
        self.assertTrue(all(case.criteria for case in benchmark.cases))

    def test_default_suite_reports_proxy_metrics_and_pass_rate(self):
        suite = run_evaluation_suite()

        self.assertEqual(suite.benchmark_id, "sourcebound-rag-fixtures-v1")
        self.assertEqual(suite.benchmark_version, "1.0.0")
        self.assertEqual(suite.case_count, 3)
        self.assertEqual(suite.pass_rate, 1.0)
        self.assertEqual(suite.mean_retrieval_recall, 1.0)
        self.assertEqual(suite.mean_claim_support, 1.0)
        self.assertEqual(suite.mean_completeness, 1.0)
        self.assertEqual(suite.metric_disclosure, FIXTURE_METRIC_DISCLOSURE)
        for result in suite.results:
            self.assertTrue(result.passed)
            self.assertGreaterEqual(result.expected_concept_count, 3)
            self.assertEqual(result.retrieval_missing_concepts, [])
            self.assertEqual(result.answer_missing_concepts, [])
            self.assertEqual(result.supported_claim_count, result.claim_count)

    def test_dataset_is_valid_json_and_matches_loaded_cases(self):
        path = Path(__file__).parents[1] / "research_agent" / "data" / "research_benchmark.json"
        payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["benchmark_id"], DEFAULT_BENCHMARK.benchmark_id)
        self.assertEqual(len(payload["cases"]), len(DEFAULT_CASES))

    def test_claim_support_metric_detects_a_mutated_unsupported_finding(self):
        report = build_offline_agent().run(DEFAULT_CASES[0].question)
        mutated_finding = report.key_findings[0].model_copy(
            update={"statement": "Quantum hardware changes ocean tides."}
        )
        mutated_report = report.model_copy(
            update={"key_findings": [mutated_finding, *report.key_findings[1:]]}
        )

        result = evaluate_report(mutated_report, DEFAULT_CASES[0])

        self.assertLess(result.claim_support, 1.0)
        self.assertFalse(result.passed)
        self.assertTrue(
            any("claim-support" in reason for reason in result.reasons)
        )


if __name__ == "__main__":
    unittest.main()
