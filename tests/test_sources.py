import unittest
from unittest.mock import patch

from research_agent.models import Source
from research_agent.retry import RetryPolicy
from research_agent.sources import (
    FixtureProvider,
    MultiSourceSearchTool,
    OpenAIWebSearchProvider,
    OpenAlexProvider,
    WikipediaProvider,
    extract_url_citations,
)


class SourceTests(unittest.TestCase):
    def test_fanout_preserves_provider_diversity_and_deduplicates_urls(self):
        shared = Source(
            id="shared-source",
            provider="provider_a",
            kind="web",
            title="Shared",
            url="https://example.com/shared",
            snippet="RAG evidence",
        )
        second = Source(
            id="second-source",
            provider="provider_b",
            kind="academic",
            title="Second",
            url="https://example.com/second",
            snippet="RAG evidence",
        )
        tool = MultiSourceSearchTool(
            [
                FixtureProvider(fixtures=[shared], source_id="provider_a"),
                FixtureProvider(
                    fixtures=[
                        shared.model_copy(update={"provider": "provider_b"}),
                        second,
                    ],
                    source_id="provider_b",
                ),
            ],
            retry_policy=RetryPolicy(max_attempts=1, jitter=0),
        )

        bundle = tool.search("RAG", limit=4)

        self.assertEqual(
            [source.url for source in bundle.sources],
            ["https://example.com/shared", "https://example.com/second"],
        )
        self.assertEqual(
            {status.provider for status in bundle.provider_status},
            {"provider_a", "provider_b"},
        )

    def test_partial_provider_failure_is_visible_without_losing_successful_sources(self):
        class BrokenProvider:
            source_id = "broken"

            def search(self, query, *, limit=5):
                raise TimeoutError("provider unavailable")

        source = Source(
            id="source-ok",
            provider="working",
            kind="web",
            title="Working source",
            url="https://example.com/working",
            snippet="Evidence",
        )
        tool = MultiSourceSearchTool(
            [
                FixtureProvider(fixtures=[source], source_id="working"),
                BrokenProvider(),
            ],
            retry_policy=RetryPolicy(max_attempts=1, jitter=0),
        )

        bundle = tool.search("topic", limit=3)

        self.assertEqual([item.id for item in bundle.sources], ["source-ok"])
        status_by_provider = {status.provider: status for status in bundle.provider_status}
        self.assertTrue(status_by_provider["working"].ok)
        self.assertFalse(status_by_provider["broken"].ok)

    def test_extracts_annotations_and_hosted_source_records(self):
        response = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "title": "Annotated source",
                                    "url": "https://example.com/a",
                                }
                            ],
                        }
                    ],
                },
                {
                    "type": "web_search_call",
                    "action": {
                        "sources": [
                            {
                                "title": "Hosted source",
                                "url": "https://example.com/b",
                            }
                        ]
                    },
                },
            ]
        }

        citations = extract_url_citations(response)

        self.assertEqual(
            [citation["url"] for citation in citations],
            ["https://example.com/a", "https://example.com/b"],
        )

    def test_hosted_web_provider_maps_citations_to_source_records(self):
        class FakeResponses:
            def __init__(self):
                self.calls = []

            def create(self, **request):
                self.calls.append(request)
                return {
                    "output": [
                        {
                            "type": "web_search_call",
                            "action": {
                                "sources": [
                                    {
                                        "title": "Hosted source",
                                        "url": "https://example.com/hosted",
                                    }
                                ]
                            },
                        }
                    ]
                }

        class FakeClient:
            def __init__(self):
                self.responses = FakeResponses()

        client = FakeClient()
        provider = OpenAIWebSearchProvider(
            client=client,
            model="test-model",
        )

        sources = provider.search("topic", limit=2)

        self.assertEqual(sources[0].provider, "openai_web_search")
        self.assertEqual(sources[0].url, "https://example.com/hosted")
        self.assertEqual(client.responses.calls[0]["tools"], [{"type": "web_search"}])
        self.assertEqual(
            client.responses.calls[0]["include"],
            ["web_search_call.action.sources"],
        )

    def test_public_provider_parsers_preserve_source_metadata(self):
        with patch(
            "research_agent.sources._http_get_json",
            return_value={
                "results": [
                    {
                        "title": "A paper",
                        "doi": "https://doi.org/10.1234/example",
                        "publication_year": 2024,
                        "authorships": [
                            {"author": {"display_name": "Researcher"}}
                        ],
                        "abstract_inverted_index": {
                            "retrieval": [0],
                            "improves": [1],
                            "grounding": [2],
                        },
                    }
                ]
            },
        ):
            sources = OpenAlexProvider().search("topic", limit=1)

        self.assertEqual(sources[0].kind, "academic")
        self.assertEqual(sources[0].published_at, "2024")
        self.assertEqual(sources[0].authors, ["Researcher"])
        self.assertIn("retrieval improves grounding", sources[0].snippet)

        with patch(
            "research_agent.sources._http_get_json",
            return_value={
                "query": {
                    "search": [
                        {
                            "title": "A topic",
                            "snippet": "<b>Useful</b> context",
                        }
                    ]
                }
            },
        ):
            sources = WikipediaProvider().search("topic", limit=1)

        self.assertEqual(sources[0].kind, "encyclopedia")
        self.assertEqual(sources[0].snippet, "Useful context")
