import unittest

from research_agent.evaluation import run_evaluation_suite


class EvaluationTests(unittest.TestCase):
    def test_fixture_suite_passes(self):
        suite = run_evaluation_suite()

        self.assertTrue(suite.passed)
        self.assertEqual(len(suite.results), 3)
        self.assertGreaterEqual(suite.mean_score, 0.8)
