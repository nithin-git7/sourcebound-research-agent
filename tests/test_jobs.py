import time
import unittest

from research_agent.jobs import JobStatus, ResearchJobManager, ResearchRequest
from research_agent.offline import build_offline_agent


class ResearchJobManagerTests(unittest.TestCase):
    def test_job_reaches_success_with_validated_report(self):
        manager = ResearchJobManager(
            lambda request: build_offline_agent().run(request.question),
            max_workers=1,
        )
        try:
            created = manager.submit(
                ResearchRequest(
                    question="What are the trade-offs of retrieval-augmented generation?",
                    mode="sample",
                )
            )
            self.assertEqual(created.status, JobStatus.QUEUED)
            deadline = time.monotonic() + 3
            current = created
            while time.monotonic() < deadline:
                current = manager.get(created.job_id)
                if current and current.status == JobStatus.SUCCEEDED:
                    break
                time.sleep(0.01)
            self.assertIsNotNone(current)
            self.assertEqual(current.status, JobStatus.SUCCEEDED)
            self.assertIsNotNone(current.report)
            self.assertEqual(current.report.question, created.question)
        finally:
            manager.close()

    def test_runner_failure_is_sanitized_into_job_state(self):
        def fail(_request):
            raise RuntimeError("provider unavailable")

        manager = ResearchJobManager(fail, max_workers=1)
        try:
            created = manager.submit(
                ResearchRequest(question="A sufficiently long research question")
            )
            deadline = time.monotonic() + 2
            current = created
            while time.monotonic() < deadline:
                current = manager.get(created.job_id)
                if current and current.status == JobStatus.FAILED:
                    break
                time.sleep(0.01)
            self.assertEqual(current.status, JobStatus.FAILED)
            self.assertEqual(current.error_code, "research_failed")
            self.assertEqual(
                current.error,
                "The research job failed before a report was produced.",
            )
            self.assertNotIn("provider unavailable", current.error)
            self.assertIsNone(current.report)
        finally:
            manager.close()


if __name__ == "__main__":
    unittest.main()
