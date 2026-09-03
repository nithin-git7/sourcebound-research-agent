import json
import unittest
from pathlib import Path

from research_agent.evaluation import (
    DEFAULT_BENCHMARK,
    DEFAULT_CASES,
    FIXTURE_METRIC_DISCLOSURE,
    RegressionThresholds,
    evaluate_report,
    evaluate_regression_thresholds,
    load_benchmark,
    run_evaluation_suite,
)
from research_agent.offline import build_offline_agent


class BenchmarkTests(unittest.TestCase):
    def test_curated_benchmark_has_v2_metadata_and_balanced_domains(self):
        benchmark = load_benchmark()
        categories = {case.category for case in benchmark.cases}
        category_counts = {
            category: sum(case.category == category for case in benchmark.cases)
            for category in categories
        }

        self.assertEqual(benchmark.mode, "fixture")
        self.assertEqual(benchmark.benchmark_id, DEFAULT_BENCHMARK.benchmark_id)
        self.assertEqual(benchmark.version, "2.0.0")
        self.assertIn("deterministic", benchmark.metric_disclosure)
        self.assertIn("not live-web quality measurements", benchmark.metric_disclosure)
        self.assertIn("not", benchmark.metric_disclosure)
        self.assertIn("semantic entailment", benchmark.metric_disclosure)
        self.assertGreaterEqual(len(benchmark.cases), 25)
        self.assertEqual(
            categories,
            {
                "technology",
                "science",
                "policy",
                "health-information-safety",
                "conflicting-source",
            },
        )
        self.assertTrue(all(count >= 5 for count in category_counts.values()))
        self.assertEqual(len({case.name for case in benchmark.cases}), len(benchmark.cases))
        self.assertEqual(len({case.question for case in benchmark.cases}), len(benchmark.cases))
        self.assertTrue(all(case.expected_concepts for case in benchmark.cases))
        self.assertTrue(all(case.fixture is not None for case in benchmark.cases))
        self.assertTrue(
            all(
                len({source.provider for source in case.fixture.sources}) >= 2
                for case in benchmark.cases
                if case.fixture is not None
            )
        )

    def test_default_suite_reports_proxy_metrics_and_pass_rate(self):
        suite = run_evaluation_suite()

        self.assertEqual(suite.benchmark_id, "sourcebound-research-benchmark-v2")
        self.assertEqual(suite.benchmark_version, "2.0.0")
        self.assertGreaterEqual(suite.case_count, 25)
        self.assertEqual(suite.pass_rate, 1.0)
        self.assertEqual(suite.mean_retrieval_recall, 1.0)
        self.assertEqual(suite.mean_claim_support, 1.0)
        self.assertEqual(suite.mean_completeness, 1.0)
        self.assertEqual(suite.metric_disclosure, DEFAULT_BENCHMARK.metric_disclosure)
        self.assertEqual(FIXTURE_METRIC_DISCLOSURE, DEFAULT_BENCHMARK.metric_disclosure)
        self.assertTrue(suite.regression_passed)
        self.assertEqual(suite.regression_failures, [])
        self.assertGreaterEqual(suite.total_evaluation_duration_ms, 0.0)
        for result in suite.results:
            self.assertTrue(result.passed)
            self.assertGreaterEqual(result.expected_concept_count, 3)
            self.assertEqual(result.retrieval_missing_concepts, [])
            self.assertEqual(result.answer_missing_concepts, [])
            self.assertEqual(result.supported_claim_count, result.claim_count)
            self.assertGreaterEqual(result.evaluation_duration_ms, 0.0)

    def test_recorded_suite_does_not_call_the_runtime_agent(self):
        def fail_if_called():
            raise AssertionError("recorded fixtures must not call the runtime agent")

        suite = run_evaluation_suite(agent_factory=fail_if_called)

        self.assertTrue(suite.passed)

    def test_regression_thresholds_report_metric_failures(self):
        passing = run_evaluation_suite().results[0]
        regressed = passing.model_copy(
            update={
                "passed": False,
                "score": 0.5,
                "retrieval_recall": 0.5,
                "claim_support": 0.5,
                "completeness": 0.5,
            }
        )

        failures = evaluate_regression_thresholds(
            [passing, regressed],
            RegressionThresholds(
                minimum_pass_rate=1.0,
                minimum_mean_score=0.8,
                minimum_mean_retrieval_recall=0.8,
                minimum_mean_claim_support=0.8,
                minimum_mean_completeness=0.8,
            ),
        )

        self.assertEqual(len(failures), 5)
        self.assertTrue(all("regression:" in failure for failure in failures))

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
