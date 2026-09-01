import unittest

from research_agent.offline import build_offline_agent
from research_agent.rendering import render_markdown
from research_agent.security import UNTRUSTED_EVIDENCE_END, UNTRUSTED_EVIDENCE_START


class RenderingTests(unittest.TestCase):
    def test_markdown_hides_internal_safety_wrappers_for_readers(self):
        report = build_offline_agent().run(
            "What are the trade-offs of retrieval-augmented generation?"
        )

        rendered = render_markdown(report)

        self.assertNotIn(UNTRUSTED_EVIDENCE_START, rendered)
        self.assertNotIn(UNTRUSTED_EVIDENCE_END, rendered)
        self.assertIn("Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", rendered)
        self.assertIn("https://arxiv.org/abs/2005.11401", rendered)


if __name__ == "__main__":
    unittest.main()
