import unittest

from research_agent.evaluation import run_evaluation_suite


class EvaluationTests(unittest.TestCase):
    def test_fixture_suite_passes(self):
        suite = run_evaluation_suite()

        self.assertTrue(suite.passed)
        self.assertGreaterEqual(len(suite.results), 25)
        self.assertGreaterEqual(suite.mean_score, 0.9)
        self.assertTrue(suite.regression_passed)
        self.assertEqual(suite.regression_failures, [])

    def test_each_required_domain_passes_its_cases(self):
        suite = run_evaluation_suite()
        required = {
            "technology",
            "science",
            "policy",
            "health-information-safety",
            "conflicting-source",
        }

        for category in required:
            results = [result for result in suite.results if result.category == category]
            self.assertGreaterEqual(len(results), 5)
            self.assertTrue(all(result.passed for result in results))
