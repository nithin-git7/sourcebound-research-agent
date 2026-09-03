import json
import unittest

from pydantic import ValidationError

from research_agent.models import ComparisonPoint, Finding, ResearchReportDraft, SourceView
from research_agent.semantic_verification import (
    SemanticEvidenceVerifier,
    SemanticJudgeOutput,
    SemanticVerdict,
    semantic_response_format,
)


def build_report(citation_ids: list[str] | None = None) -> ResearchReportDraft:
    return ResearchReportDraft(
        executive_summary="Summary.",
        key_findings=[
            Finding(
                finding_id="finding-1",
                statement="Retrieval improves answer freshness.",
                importance="high",
                confidence=0.9,
                citation_ids=citation_ids or ["source-a"],
            )
        ],
        comparison=[
            ComparisonPoint(
                dimension="freshness",
                consensus="Retrieval can use current evidence.",
                disagreements=[],
                source_views=[SourceView(source_id="source-a", position="Supports retrieval.")],
            )
        ],
        limitations=["Small corpus."],
    )


EVIDENCE = {
    "source-a": {
        "title": "Retrieval study",
        "snippet": "Retrieval supplies current external evidence.",
        "url": "https://example.com/a",
        "provider": "academic",
    }
}


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def response(
    verdict: str,
    *,
    confidence: float = 0.88,
    confidence_basis: str = "direct",
    supporting: list[str] | None = None,
    conflicting: list[str] | None = None,
):
    return {
        "output_text": json.dumps(
            {
                "verdict": verdict,
                "confidence": confidence,
                "confidence_basis": confidence_basis,
                "rationale": "The supplied evidence determines this verdict.",
                "supporting_citation_ids": supporting or [],
                "conflicting_citation_ids": conflicting or [],
            }
        )
    }


class SemanticVerificationTests(unittest.TestCase):
    def test_contract_and_response_schema_are_strict(self):
        with self.assertRaises(ValidationError):
            SemanticJudgeOutput(
                verdict="supported",
                confidence=0.8,
                confidence_basis="direct",
                rationale="Grounded.",
                supporting_citation_ids=["source-a"],
                conflicting_citation_ids=[],
                unexpected=True,
            )

        schema = semantic_response_format()
        self.assertTrue(schema["strict"])
        self.assertFalse(schema["schema"]["additionalProperties"])
        self.assertEqual(set(schema["schema"]["required"]), set(schema["schema"]["properties"]))

    def test_uses_evidence_only_responses_style_request(self):
        client = FakeClient(
            [
                response("supported", supporting=["source-a"]),
                response("partial", confidence_basis="partial", supporting=["source-a"]),
            ]
        )
        result = SemanticEvidenceVerifier(client, model="judge-model").verify(build_report(), EVIDENCE)

        self.assertEqual(result.supported_claim_count, 1)
        self.assertEqual(result.partial_claim_count, 1)
        request = client.requests[0]
        self.assertEqual(request["model"], "judge-model")
        self.assertIn("ONLY the supplied EVIDENCE", request["input"][0]["content"])
        self.assertNotIn("tools", request)
        user_payload = json.loads(request["input"][1]["content"])
        self.assertEqual(user_payload["evidence"][0]["citation_id"], "source-a")
        self.assertNotIn("url", user_payload["evidence"][0])
        self.assertEqual(request["text"]["format"]["type"], "json_schema")

    def test_all_five_verdicts_are_supported(self):
        verdicts = [
            "supported",
            "partial",
            "unsupported",
            "contradicted",
            "insufficient_evidence",
        ]
        for verdict in verdicts:
            with self.subTest(verdict=verdict):
                basis = {
                    "partial": "partial",
                    "unsupported": "absence",
                    "contradicted": "conflict",
                    "insufficient_evidence": "absence",
                }.get(verdict, "direct")
                client = FakeClient(
                    [response(verdict, confidence_basis=basis), response(verdict, confidence_basis=basis)]
                )
                result = SemanticEvidenceVerifier(client).verify(build_report(), EVIDENCE)
                self.assertEqual(result.claim_checks[0].verdict, SemanticVerdict(verdict))

    def test_confidence_is_calibrated_by_evidence_basis(self):
        client = FakeClient(
            [
                response("partial", confidence=0.99, confidence_basis="partial", supporting=["source-a"]),
                response("unsupported", confidence=0.99, confidence_basis="absence"),
            ]
        )
        result = SemanticEvidenceVerifier(client).verify(build_report(), EVIDENCE)

        self.assertEqual(result.claim_checks[0].confidence, 0.74)
        self.assertEqual(result.claim_checks[1].confidence, 0.65)

    def test_model_and_validation_failures_are_deterministic(self):
        client = FakeClient([RuntimeError("secret provider detail"), {"output_text": "not json"}])
        result = SemanticEvidenceVerifier(client).verify(build_report(), EVIDENCE)

        first, second = result.claim_checks
        self.assertEqual(first.verdict, SemanticVerdict.INSUFFICIENT_EVIDENCE)
        self.assertEqual(first.error.code, "model_error")
        self.assertEqual(first.confidence, 0.0)
        self.assertNotIn("secret", first.error.message)
        self.assertEqual(second.error.code, "invalid_response")
        self.assertEqual(result.error_count, 2)

    def test_out_of_scope_model_citations_fail_closed(self):
        client = FakeClient(
            [
                response("supported", supporting=["invented-source"]),
                response("supported", supporting=["invented-source"]),
            ]
        )
        result = SemanticEvidenceVerifier(client).verify(build_report(), EVIDENCE)

        self.assertTrue(all(check.error.code == "invalid_citation_reference" for check in result.claim_checks))
        self.assertTrue(
            all(check.verdict is SemanticVerdict.INSUFFICIENT_EVIDENCE for check in result.claim_checks)
        )

    def test_missing_cited_evidence_skips_the_client(self):
        client = FakeClient([])
        report = build_report(["missing-source"])
        # The comparison also cites source-a, so omit all evidence to make both
        # claim checks fail closed before any model call.
        result = SemanticEvidenceVerifier(client).verify(report, {})

        self.assertEqual(client.requests, [])
        self.assertEqual(result.error_count, 2)
        self.assertEqual(result.insufficient_evidence_claim_count, 2)
        self.assertTrue(all(check.confidence == 0.0 for check in result.claim_checks))


if __name__ == "__main__":
    unittest.main()
