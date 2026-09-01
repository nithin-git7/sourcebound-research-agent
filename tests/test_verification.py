import unittest

from pydantic import ValidationError

from research_agent.models import (
    ComparisonPoint,
    EvidenceAudit,
    Finding,
    ProviderStatus,
    ResearchReport,
    ResearchReportDraft,
    Source,
    SourceView,
    ToolCallTrace,
)
from research_agent.verification import (
    ClaimReference,
    DeterministicLexicalVerifier,
    EvidenceVerdict,
    extract_claim_references,
    verify_evidence,
)


def build_report(*, statement: str = "Retrieval improves answer freshness") -> ResearchReportDraft:
    return ResearchReportDraft(
        executive_summary="A concise summary.",
        key_findings=[
            Finding(
                finding_id="finding-1",
                statement=statement,
                importance="high",
                confidence=0.9,
                citation_ids=["source-a", "source-b"],
            )
        ],
        comparison=[
            ComparisonPoint(
                dimension="freshness",
                consensus="The sources discuss retrieval and freshness.",
                disagreements=["The sources differ on operational complexity."],
                source_views=[
                    SourceView(source_id="source-a", position="Discusses retrieval."),
                    SourceView(source_id="source-b", position="Discusses freshness."),
                ],
            )
        ],
        limitations=["Small test corpus."],
    )


EVIDENCE = {
    "source-a": {
        "title": "Retrieval quality study",
        "snippet": "Retrieval improves answer freshness by supplying current external evidence.",
        "url": "https://example.com/a",
        "provider": "academic",
    },
    "source-b": {
        "title": "Production retrieval notes",
        "snippet": "Retrieval systems add operational complexity and latency.",
        "url": "https://example.com/b",
        "provider": "engineering",
    },
}


class VerificationTests(unittest.TestCase):
    def test_contracts_are_strict_and_closed(self):
        with self.assertRaises(ValidationError):
            ClaimReference(
                claim_id="claim-1",
                statement="A claim",
                citation_ids=["source-a"],
                claim_type="finding",
                unexpected=True,
            )
        with self.assertRaises(ValidationError):
            ClaimReference(
                claim_id=1,
                statement="A claim",
                citation_ids=["source-a"],
                claim_type="finding",
            )

    def test_extracts_findings_and_comparison_assertions(self):
        references = extract_claim_references(build_report())

        self.assertEqual(
            [reference.claim_id for reference in references],
            [
                "finding-1",
                "comparison-1-consensus",
                "comparison-1-disagreement-1",
            ],
        )
        self.assertEqual(references[0].citation_ids, ["source-a", "source-b"])
        self.assertEqual(references[1].claim_type, "comparison_consensus")

    def test_extracts_the_same_claims_from_a_final_report(self):
        draft = build_report()
        final_report = ResearchReport.create(
            run_id="run-1",
            question="A research question",
            draft=draft,
            sources=[
                Source(
                    id="source-a",
                    provider="academic",
                    title="A",
                    url="https://example.com/a",
                    snippet="Evidence.",
                )
            ],
            provider_status=[ProviderStatus(provider_id="academic", ok=True, result_count=1)],
            audit=EvidenceAudit(
                citation_coverage=1.0,
                grounding_score=1.0,
                source_diversity=0.5,
                comparison_quality=0.5,
                score=0.8,
                cited_finding_count=1,
                finding_count=1,
                provider_count=1,
            ),
            model="test-model",
            tool_calls=[
                ToolCallTrace(
                    name="search_sources",
                    query="A research question",
                    source_count=1,
                    providers=["academic"],
                )
            ],
        )

        self.assertEqual(
            [reference.claim_id for reference in extract_claim_references(final_report)],
            [reference.claim_id for reference in extract_claim_references(draft)],
        )

    def test_supported_claim_returns_matched_evidence(self):
        result = verify_evidence(build_report(), EVIDENCE)
        finding = result.claim_checks[0]

        self.assertEqual(finding.verdict, EvidenceVerdict.SUPPORTED)
        self.assertEqual(finding.invalid_citation_ids, [])
        self.assertTrue(finding.matched_evidence)
        self.assertIn("retrieval improves", finding.matched_evidence[0].matched_phrases)
        self.assertGreaterEqual(finding.coverage, 0.65)
        self.assertTrue(any("not semantic entailment" in warning for warning in result.warnings))

    def test_partial_and_unsupported_claims_are_visible(self):
        report = build_report(statement="Retrieval affects answer latency")
        result = DeterministicLexicalVerifier().verify(report, {"source-a": EVIDENCE["source-a"]})
        finding = result.claim_checks[0]

        self.assertEqual(finding.verdict, EvidenceVerdict.PARTIAL)
        self.assertEqual(result.citation_coverage, 0.5)
        self.assertIn("source-b", finding.invalid_citation_ids)
        self.assertTrue(any("Unknown citation IDs" in warning for warning in finding.warnings))

        unsupported = verify_evidence(
            build_report(statement="Quantum hardware changes ocean tides"),
            EVIDENCE,
        )
        self.assertEqual(unsupported.claim_checks[0].verdict, EvidenceVerdict.UNSUPPORTED)
        self.assertGreater(unsupported.unsupported_claim_count, 0)
        self.assertTrue(any("No meaningful lexical evidence" in warning for warning in unsupported.warnings))

    def test_explicit_negation_is_conservatively_flagged_as_contradiction(self):
        report = build_report(statement="Retrieval does not improve answer freshness")
        result = verify_evidence(report, {"source-a": EVIDENCE["source-a"]})

        finding = result.claim_checks[0]
        self.assertEqual(finding.verdict, EvidenceVerdict.CONTRADICTED)
        self.assertEqual(result.contradicted_claim_count, 1)
        self.assertTrue(any("review manually" in warning for warning in finding.warnings))

    def test_malformed_records_are_skipped_and_reported(self):
        result = verify_evidence(build_report(), {"source-a": {"title": "Missing text"}})

        self.assertEqual(result.citation_coverage, 0.0)
        self.assertTrue(any("Skipped malformed evidence" in warning for warning in result.warnings))
        self.assertTrue(result.claim_checks[0].invalid_citation_ids)


if __name__ == "__main__":
    unittest.main()
