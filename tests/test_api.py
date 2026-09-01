import importlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_REPORT = ROOT / "portfolio" / "sample_report.json"


class ApiTests(unittest.TestCase):
    def test_api_module_import_is_lazy(self):
        module = importlib.import_module("research_agent.api")

        self.assertTrue(callable(module.create_app))
        self.assertNotIn("FastAPI", module.__dict__)

    def test_health_and_report_endpoints(self):
        try:
            from fastapi.testclient import TestClient
        except ModuleNotFoundError:
            self.skipTest("FastAPI is not installed; optional API test skipped.")

        from research_agent.api import create_app

        app = create_app(report_path=SAMPLE_REPORT)
        client = TestClient(app)

        health = client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")
        self.assertEqual(health.json()["service"], "sourcebound-research-agent")

        response = client.get("/report")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["run_id"], "portfolio-demo-001")
        self.assertEqual(payload["question"], "What are the trade-offs of retrieval-augmented generation?")
        self.assertEqual(len(payload["sources"]), 4)
        self.assertEqual(payload["tool_calls"][0]["stop_reason"], "coverage_met")

        alias = client.get("/api/report")
        self.assertEqual(alias.status_code, 200)
        self.assertEqual(alias.json(), payload)

    def test_report_can_be_supplied_as_a_mapping(self):
        try:
            from fastapi.testclient import TestClient
        except ModuleNotFoundError:
            self.skipTest("FastAPI is not installed; optional API test skipped.")

        from research_agent.api import create_app

        payload = json.loads(SAMPLE_REPORT.read_text(encoding="utf-8"))
        client = TestClient(create_app(report=payload))

        response = client.get("/report")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["run_id"], "portfolio-demo-001")

    def test_missing_report_uses_a_validated_offline_fallback(self):
        try:
            from fastapi.testclient import TestClient
        except ModuleNotFoundError:
            self.skipTest("FastAPI is not installed; optional API test skipped.")

        from research_agent.api import create_app

        missing_path = ROOT / "portfolio" / "missing-report-for-test.json"
        response = TestClient(create_app(report_path=missing_path)).get("/report")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["run_id"], "portfolio-demo-001")
        self.assertGreaterEqual(len(payload["sources"]), 1)


if __name__ == "__main__":
    unittest.main()
