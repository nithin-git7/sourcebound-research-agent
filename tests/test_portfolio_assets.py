import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PortfolioAssetTests(unittest.TestCase):
    def test_requested_portfolio_and_release_files_exist(self):
        paths = [
            ROOT / "research_agent" / "api.py",
            ROOT / "portfolio" / "index.html",
            ROOT / "portfolio" / "styles.css",
            ROOT / "portfolio" / "app.js",
            ROOT / "portfolio" / "sample_report.json",
            ROOT / ".github" / "workflows" / "ci.yml",
            ROOT / "Dockerfile",
            ROOT / ".dockerignore",
        ]

        for path in paths:
            with self.subTest(path=path):
                self.assertTrue(path.is_file(), path)
                self.assertGreater(path.stat().st_size, 0, path)

    def test_sample_report_is_deterministic_and_contract_valid(self):
        path = ROOT / "portfolio" / "sample_report.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        from research_agent.models import ResearchReport

        ResearchReport.model_validate_json(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["run_id"], "portfolio-demo-001")
        self.assertEqual(payload["generated_at"], "2026-01-15T09:00:00+00:00")
        self.assertEqual(len(payload["sources"]), 4)
        self.assertEqual(len(payload["tool_calls"][0]["planned_queries"]), 3)
        self.assertEqual(payload["tool_calls"][0]["stop_reason"], "coverage_met")
        self.assertEqual(payload["verification"]["method"], "deterministic_lexical")
        self.assertEqual(
            {source["id"] for source in payload["sources"]},
            {"src-academic-rag", "src-survey-rag", "src-production-rag", "src-evaluation-rag"},
        )

    def test_viewer_is_local_accessible_and_stateful(self):
        html = (ROOT / "portfolio" / "index.html").read_text(encoding="utf-8").lower()
        javascript = (ROOT / "portfolio" / "app.js").read_text(encoding="utf-8").lower()
        css = (ROOT / "portfolio" / "styles.css").read_text(encoding="utf-8").lower()

        for marker in [
            'id="trace-viewer"',
            'id="trace-nav"',
            'id="trace-stage"',
            'id="trace-prev"',
            'id="trace-next"',
            'id="trace-retry"',
        ]:
            self.assertIn(marker, html)
        for stage in ["question", "planned queries", "providers", "evidence", "claims", "verification"]:
            self.assertIn(stage, html)
        for state in ["loading", "empty", "error"]:
            self.assertIn(state, javascript)
        self.assertIn("sample_report.json", javascript)
        self.assertIn("aria-live", html)
        self.assertNotIn("cdn.", html + javascript + css)
        self.assertNotIn("fonts.googleapis", html + javascript + css)
        self.assertIn("#c8ff52", css)
        self.assertIn("@media (max-width: 680px)", css)
        self.assertIn("prefers-reduced-motion", css)

    def test_release_artifacts_are_truthful_and_complete(self):
        workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
        api = (ROOT / "research_agent" / "api.py").read_text(encoding="utf-8")

        self.assertIn("python -m unittest discover -v", workflow)
        self.assertIn("python -m compileall -q research_agent tests", workflow)
        self.assertIn("python -m pip wheel . --no-deps --wheel-dir dist", workflow)
        self.assertIn("python-version: ${{ matrix.python-version }}", workflow)
        self.assertNotIn("\\${{", workflow)
        self.assertIn(".[live,web]", dockerfile)
        package = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn("web =", package)
        self.assertIn("fastapi", package)
        self.assertIn("uvicorn", package)
        self.assertIn("COPY portfolio ./portfolio", dockerfile)
        self.assertIn("CMD [\"uvicorn\"", dockerfile)
        self.assertIn("tests", dockerignore)
        self.assertIn("def create_app", api)
        self.assertIn("def health", api)
        self.assertIn("def report_endpoint", api)

    def test_portfolio_copy_respects_visual_content_contract(self):
        html = (ROOT / "portfolio" / "index.html").read_text(encoding="utf-8")
        javascript = (ROOT / "portfolio" / "app.js").read_text(encoding="utf-8")

        self.assertNotRegex(html, r"[\u2013\u2014]")
        self.assertNotRegex(javascript, r"[\u2013\u2014]")
        self.assertNotIn('class="principle-number"', html)
        self.assertNotIn('Case study <span>/</span> 07', html)
        self.assertNotIn('<p class="eyebrow">02', html)


if __name__ == "__main__":
    unittest.main()
