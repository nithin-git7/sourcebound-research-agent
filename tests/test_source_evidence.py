import unittest
from dataclasses import dataclass
from unittest.mock import patch

from research_agent.models import Source
from research_agent.sources import OpenAIWebSearchProvider, extract_url_citations


def _dict_response() -> dict:
    text = "Alpha evidence supports retrieval quality [A]. Beta evidence warns about latency [B]."
    alpha_start = text.index("[A]")
    beta_start = text.index("[B]")
    return {
        "output_text": text,
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": [
                            {
                                "type": "url_citation",
                                "title": "Alpha source",
                                "url": "https://example.com/alpha",
                                "start_index": alpha_start,
                                "end_index": alpha_start + 3,
                            },
                            {
                                "type": "url_citation",
                                "title": "Beta source",
                                "url": "https://example.com/beta",
                                "start_index": beta_start,
                                "end_index": beta_start + 3,
                            },
                        ],
                    }
                ],
            },
            {
                "type": "web_search_call",
                "action": {
                    "sources": [
                        {
                            "type": "url",
                            "title": "Alpha source",
                            "url": "https://example.com/alpha",
                            "snippet": "Alpha source excerpt about retrieval quality.",
                            "metadata": {
                                "publisher": "Alpha Press",
                                "kind": "article",
                                "published_at": "2025-01-02",
                            },
                        },
                        {
                            "type": "url",
                            "title": "Beta source",
                            "url": "https://example.com/beta",
                            "snippet": "Beta source excerpt about latency trade-offs.",
                            "metadata": {
                                "publisher": "Beta Lab",
                                "kind": "report",
                                "authors": ["B. Analyst"],
                                "credibility": "high",
                            },
                        },
                    ]
                },
            },
        ],
    }


class SourceEvidenceTests(unittest.TestCase):
    def test_dict_fixture_keeps_distinct_evidence_offsets_and_metadata(self):
        citations = extract_url_citations(_dict_response())

        self.assertEqual([item["url"] for item in citations], [
            "https://example.com/alpha",
            "https://example.com/beta",
        ])
        self.assertEqual(citations[0]["snippet"], "Alpha source excerpt about retrieval quality.")
        self.assertEqual(citations[1]["snippet"], "Beta source excerpt about latency trade-offs.")
        self.assertNotEqual(citations[0]["evidence"], citations[1]["evidence"])
        self.assertEqual(citations[0]["start_index"], _dict_response()["output_text"].index("[A]"))
        self.assertEqual(citations[1]["end_index"], _dict_response()["output_text"].index("[B]") + 3)
        self.assertEqual(citations[0]["metadata"]["publisher"], "Alpha Press")
        self.assertEqual(citations[1]["metadata"]["authors"], ["B. Analyst"])

    def test_sdk_objects_map_source_specific_evidence_and_supported_metadata(self):
        @dataclass
        class Annotation:
            type: str
            title: str
            url: str
            start_index: int
            end_index: int

        @dataclass
        class OutputText:
            type: str
            text: str
            annotations: list[Annotation]

        @dataclass
        class Content:
            type: str
            text: OutputText

        @dataclass
        class Message:
            type: str
            content: list[Content]

        @dataclass
        class HostedSource:
            type: str
            title: str
            url: str
            snippet: str
            metadata: dict

        @dataclass
        class Action:
            sources: list[HostedSource]

        @dataclass
        class WebSearchCall:
            type: str
            action: Action

        @dataclass
        class Response:
            output_text: str
            output: list[object]

        text = "First source explains recall. Second source explains cost."
        response = Response(
            output_text=text,
            output=[
                Message(
                    type="message",
                    content=[
                        Content(
                            type="output_text",
                            text=OutputText(
                                type="output_text",
                                text=text,
                                annotations=[
                                    Annotation("url_citation", "First", "https://example.com/first", 0, 5),
                                    Annotation("url_citation", "Second", "https://example.com/second", 34, 40),
                                ],
                            ),
                        )
                    ],
                ),
                WebSearchCall(
                    type="web_search_call",
                    action=Action(
                        sources=[
                            HostedSource(
                                "url",
                                "First",
                                "https://example.com/first",
                                "First source-specific excerpt.",
                                {"kind": "article", "published_at": "2024", "publisher": "First Press"},
                            ),
                            HostedSource(
                                "url",
                                "Second",
                                "https://example.com/second",
                                "Second source-specific excerpt.",
                                {"kind": "report", "authors": ["Second Author"], "credibility": "high"},
                            ),
                        ]
                    ),
                ),
            ],
        )

        class FakeResponses:
            def create(self, **request):
                return response

        class FakeClient:
            responses = FakeResponses()

        sources = OpenAIWebSearchProvider(client=FakeClient()).search("topic", limit=5)

        self.assertIsInstance(sources[0], Source)
        self.assertEqual([source.snippet for source in sources], [
            "First source-specific excerpt.",
            "Second source-specific excerpt.",
        ])
        self.assertEqual(sources[0].kind, "article")
        self.assertEqual(sources[0].published_at, "2024")
        self.assertEqual(sources[1].authors, ["Second Author"])
        self.assertEqual(sources[1].credibility, "high")
        self.assertEqual(sources[0].evidence_text, "First source-specific excerpt.")
        self.assertEqual(sources[0].start_index, 0)
        self.assertEqual(sources[0].end_index, 5)
        self.assertEqual(sources[0].metadata["publisher"], "First Press")

    def test_optional_provenance_fields_forward_when_supported_by_source_model(self):
        @dataclass
        class RichSource:
            id: str
            provider: str
            title: str
            url: str
            snippet: str
            evidence: str = ""
            start_index: int | None = None
            end_index: int | None = None
            metadata: dict | None = None

        class FakeResponses:
            def create(self, **request):
                return _dict_response()

        class FakeClient:
            responses = FakeResponses()

        with patch("research_agent.sources.Source", RichSource):
            sources = OpenAIWebSearchProvider(client=FakeClient()).search("topic", limit=2)

        self.assertEqual(sources[0].evidence, "Alpha source excerpt about retrieval quality.")
        self.assertEqual(sources[1].evidence, "Beta source excerpt about latency trade-offs.")
        self.assertEqual(sources[0].start_index, _dict_response()["output_text"].index("[A]"))
        self.assertEqual(sources[1].end_index, _dict_response()["output_text"].index("[B]") + 3)
        self.assertEqual(sources[0].metadata["publisher"], "Alpha Press")


if __name__ == "__main__":
    unittest.main()
