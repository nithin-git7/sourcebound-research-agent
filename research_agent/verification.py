"""Provider-agnostic, deterministic claim-to-evidence verification.

This module deliberately performs lexical checks only. A lexical overlap is
useful for offline regression tests and for surfacing missing evidence, but it
is not a semantic entailment judgment. The optional evidence-only semantic
verifier is exposed separately without weakening the deterministic default.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from typing import Any, Protocol, TypeAlias

from pydantic import Field, StrictFloat, StrictInt, StrictStr

from .models import ResearchReport, ResearchReportDraft, Source, StrictModel


ReportLike: TypeAlias = ResearchReportDraft | ResearchReport
EvidenceInput: TypeAlias = Mapping[str, Any] | Iterable[Any]


class EvidenceVerdict(str, Enum):
    """The deterministic status assigned to one claim."""

    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"


class ClaimReference(StrictModel):
    """One report assertion and the source IDs that the report attached to it."""

    claim_id: StrictStr = Field(min_length=1, max_length=120)
    statement: StrictStr = Field(min_length=1, max_length=2_000)
    citation_ids: list[StrictStr] = Field(min_length=1, max_length=12)
    claim_type: StrictStr = Field(min_length=1, max_length=40)


class EvidenceRecord(StrictModel):
    """Normalized evidence accepted from any provider implementation."""

    citation_id: StrictStr = Field(min_length=1, max_length=120)
    title: StrictStr = Field(min_length=1, max_length=500)
    text: StrictStr = Field(min_length=1, max_length=10_000)
    source_url: StrictStr | None = Field(default=None, max_length=2_000)
    provider: StrictStr | None = Field(default=None, max_length=120)


class MatchedEvidence(StrictModel):
    """A source record with the exact lexical material that matched a claim."""

    citation_id: StrictStr = Field(min_length=1, max_length=120)
    source_title: StrictStr = Field(min_length=1, max_length=500)
    evidence_text: StrictStr = Field(min_length=1, max_length=2_000)
    matched_phrases: list[StrictStr] = Field(default_factory=list, max_length=12)
    lexical_score: StrictFloat = Field(ge=0.0, le=1.0)


class ClaimEvidenceCheck(StrictModel):
    """Verification result for one report claim."""

    claim_id: StrictStr = Field(min_length=1, max_length=120)
    statement: StrictStr = Field(min_length=1, max_length=2_000)
    citation_ids: list[StrictStr] = Field(min_length=1, max_length=12)
    valid_citation_ids: list[StrictStr] = Field(default_factory=list, max_length=12)
    invalid_citation_ids: list[StrictStr] = Field(default_factory=list, max_length=12)
    verdict: EvidenceVerdict
    matched_evidence: list[MatchedEvidence] = Field(default_factory=list, max_length=12)
    coverage: StrictFloat = Field(ge=0.0, le=1.0)
    warnings: list[StrictStr] = Field(default_factory=list, max_length=12)
    method: StrictStr = Field(default="deterministic_lexical", min_length=1, max_length=80)


class EvidenceVerificationReport(StrictModel):
    """Aggregate verification output for a draft or final report."""

    claim_checks: list[ClaimEvidenceCheck] = Field(default_factory=list, max_length=64)
    citation_coverage: StrictFloat = Field(ge=0.0, le=1.0)
    claim_coverage: StrictFloat = Field(ge=0.0, le=1.0)
    supported_claim_count: StrictInt = Field(ge=0)
    partial_claim_count: StrictInt = Field(ge=0)
    unsupported_claim_count: StrictInt = Field(ge=0)
    contradicted_claim_count: StrictInt = Field(ge=0)
    warnings: list[StrictStr] = Field(default_factory=list, max_length=128)
    method: StrictStr = Field(default="deterministic_lexical", min_length=1, max_length=80)


class EvidenceVerifier(Protocol):
    """Extension point for deterministic or future judge implementations."""

    def verify(
        self,
        report: ReportLike,
        evidence: EvidenceInput,
    ) -> EvidenceVerificationReport:
        """Verify report claims against provider-returned evidence."""


class EvidenceContractError(ValueError):
    """Raised when an evidence record cannot be normalized safely."""


_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
_NEGATION_RE = re.compile(
    r"\b(?:not|no|never|cannot|can't|doesn't|does not|without|fails to|lack(?:s|ed)?|unlikely)\b",
    re.IGNORECASE,
)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "may",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "with",
    "which",
    "without",
}


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in (match.group(0).lower() for match in _TOKEN_RE.finditer(text))
        if len(token) > 1 and token not in _STOPWORDS
    ]


def _ngrams(tokens: Sequence[str], size: int) -> set[tuple[str, ...]]:
    return {
        tuple(tokens[index : index + size])
        for index in range(max(0, len(tokens) - size + 1))
    }


def _matched_phrases(claim_tokens: Sequence[str], evidence_tokens: Sequence[str]) -> list[str]:
    matches: list[tuple[int, str]] = []
    for size in (4, 3, 2):
        for phrase in sorted(_ngrams(claim_tokens, size)):
            if phrase in _ngrams(evidence_tokens, size):
                matches.append((size, " ".join(phrase)))
    if matches:
        return [phrase for _, phrase in sorted(matches, key=lambda item: (-item[0], item[1]))[:12]]
    overlap = sorted(set(claim_tokens) & set(evidence_tokens))
    return overlap[:12]


def _negated(text: str) -> bool:
    return bool(_NEGATION_RE.search(text))


def _read(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _as_evidence_record(item: Any, fallback_id: str | None = None) -> EvidenceRecord:
    if isinstance(item, EvidenceRecord):
        return item

    citation_id = _read(item, "citation_id")
    if citation_id is None:
        citation_id = _read(item, "source_id")
    if citation_id is None:
        citation_id = _read(item, "id", fallback_id)

    title = _read(item, "title", "Untitled source")
    # Hosted search records can retain a source-specific evidence excerpt in
    # evidence_text; prefer it over any generic response text when present.
    text = _read(item, "evidence_text")
    if text is None:
        text = _read(item, "evidence")
    if text is None:
        text = _read(item, "text")
    if text is None:
        text = _read(item, "snippet")
    if text is None:
        text = _read(item, "content")
    if text is None:
        text = _read(item, "abstract")

    if not isinstance(citation_id, str) or not isinstance(title, str) or not isinstance(text, str):
        raise EvidenceContractError("Evidence requires string citation ID, title, and text.")
    if not citation_id.strip() or not title.strip() or not text.strip():
        raise EvidenceContractError("Evidence citation ID, title, and text cannot be blank.")

    return EvidenceRecord(
        citation_id=citation_id,
        title=title,
        text=text,
        source_url=_read(item, "source_url", _read(item, "url")),
        provider=_read(item, "provider"),
    )


def _normalize_evidence(evidence: EvidenceInput) -> tuple[dict[str, EvidenceRecord], list[str]]:
    records: dict[str, EvidenceRecord] = {}
    warnings: list[str] = []
    items: Iterable[tuple[str | None, Any]]
    if isinstance(evidence, Mapping):
        items = ((str(key), value) for key, value in evidence.items())
    else:
        items = ((None, value) for value in evidence)

    for fallback_id, item in items:
        try:
            record = _as_evidence_record(item, fallback_id=fallback_id)
        except (EvidenceContractError, TypeError, ValueError) as exc:
            warnings.append(f"Skipped malformed evidence record: {exc}")
            continue
        if record.citation_id in records:
            warnings.append(f"Duplicate evidence ID ignored: {record.citation_id}")
            continue
        records[record.citation_id] = record
    return records, warnings


def extract_claim_references(report: ReportLike) -> list[ClaimReference]:
    """Extract finding and comparison assertions from either report shape."""

    references: list[ClaimReference] = []
    for finding in report.key_findings:
        references.append(
            ClaimReference(
                claim_id=finding.finding_id,
                statement=finding.statement,
                citation_ids=list(dict.fromkeys(finding.citation_ids)),
                claim_type="finding",
            )
        )

    for index, comparison in enumerate(report.comparison, start=1):
        citation_ids = list(dict.fromkeys(view.source_id for view in comparison.source_views))
        references.append(
            ClaimReference(
                claim_id=f"comparison-{index}-consensus",
                statement=comparison.consensus,
                citation_ids=citation_ids,
                claim_type="comparison_consensus",
            )
        )
        for disagreement_index, disagreement in enumerate(comparison.disagreements, start=1):
            references.append(
                ClaimReference(
                    claim_id=f"comparison-{index}-disagreement-{disagreement_index}",
                    statement=disagreement,
                    citation_ids=citation_ids,
                    claim_type="comparison_disagreement",
                )
            )
    return references


def _score_claim(statement: str, evidence: EvidenceRecord) -> tuple[float, list[str]]:
    claim_tokens = _tokens(statement)
    evidence_tokens = _tokens(f"{evidence.title} {evidence.text}")
    if not claim_tokens or not evidence_tokens:
        return 0.0, []
    overlap = set(claim_tokens) & set(evidence_tokens)
    token_recall = len(overlap) / len(set(claim_tokens))
    phrase_matches = _matched_phrases(claim_tokens, evidence_tokens)
    phrase_score = min(1.0, len(phrase_matches) / max(1, min(3, len(claim_tokens) - 1)))
    score = min(1.0, 0.7 * token_recall + 0.3 * phrase_score)
    return round(score, 4), phrase_matches


def _check_claim(
    claim: ClaimReference,
    evidence_by_id: Mapping[str, EvidenceRecord],
) -> ClaimEvidenceCheck:
    valid_ids = [citation_id for citation_id in claim.citation_ids if citation_id in evidence_by_id]
    invalid_ids = [citation_id for citation_id in claim.citation_ids if citation_id not in evidence_by_id]
    warnings: list[str] = []
    if invalid_ids:
        warnings.append("Unknown citation IDs: " + ", ".join(invalid_ids) + ".")

    matches: list[MatchedEvidence] = []
    scores: list[float] = []
    contradiction_flags: list[bool] = []
    for citation_id in valid_ids:
        source = evidence_by_id[citation_id]
        score, phrases = _score_claim(claim.statement, source)
        scores.append(score)
        # A polarity mismatch is only meaningful when the same source has a
        # strong lexical match. This avoids treating an unrelated source's use
        # of words such as "without" as a contradiction of the whole claim.
        contradiction = (
            score >= 0.45
            and _negated(claim.statement) != _negated(source.text)
            and any(len(phrase.split()) >= 2 for phrase in phrases)
        )
        contradiction_flags.append(contradiction)
        if score > 0:
            matches.append(
                MatchedEvidence(
                    citation_id=citation_id,
                    source_title=source.title,
                    evidence_text=source.text[:2_000],
                    matched_phrases=phrases,
                    lexical_score=score,
                )
            )

    # Claims may cite multiple sources, so score the union of their evidence as
    # well as each source independently. The union lets two complementary
    # citations support a multi-clause finding without pretending either source
    # supports every clause alone.
    combined_sources = [evidence_by_id[citation_id] for citation_id in valid_ids]
    combined_text = " ".join(
        f"{source.title} {source.text}" for source in combined_sources
    )
    combined_record = EvidenceRecord(
        citation_id="combined",
        title="Combined cited evidence",
        text=combined_text or "No cited evidence",
    )
    aggregate_score, _ = _score_claim(claim.statement, combined_record)
    strongest_score = max(aggregate_score, max(scores, default=0.0))
    has_contradiction = any(contradiction_flags)
    # This threshold is intentionally modest because source snippets are often
    # shorter than the synthesized claim. The matched passage is still exposed
    # and the report labels the method as lexical rather than semantic.
    has_support = strongest_score >= 0.50 and not invalid_ids
    has_partial = strongest_score >= 0.25
    if has_contradiction and not has_support:
        verdict = EvidenceVerdict.CONTRADICTED
        warnings.append("Explicit negation conflicts with the strongest lexical match; review manually.")
    elif has_contradiction and has_support:
        verdict = EvidenceVerdict.PARTIAL
        warnings.append("Sources contain conflicting explicit-negation cues; review manually.")
    elif has_support:
        verdict = EvidenceVerdict.SUPPORTED
    elif has_partial:
        verdict = EvidenceVerdict.PARTIAL
        warnings.append("Only partial lexical overlap was found; semantic support is unverified.")
    else:
        verdict = EvidenceVerdict.UNSUPPORTED
        warnings.append("No meaningful lexical evidence was found for this claim.")

    if not valid_ids:
        warnings.append("The claim has no resolvable evidence records.")
    coverage = strongest_score if valid_ids else 0.0
    return ClaimEvidenceCheck(
        claim_id=claim.claim_id,
        statement=claim.statement,
        citation_ids=claim.citation_ids,
        valid_citation_ids=valid_ids,
        invalid_citation_ids=invalid_ids,
        verdict=verdict,
        matched_evidence=matches,
        coverage=round(coverage, 4),
        warnings=warnings,
    )


class DeterministicLexicalVerifier:
    """Offline verifier based on token overlap, phrases, and explicit negation cues."""

    method = "deterministic_lexical"

    def verify(
        self,
        report: ReportLike,
        evidence: EvidenceInput,
    ) -> EvidenceVerificationReport:
        claims = extract_claim_references(report)
        evidence_by_id, normalization_warnings = _normalize_evidence(evidence)
        checks = [_check_claim(claim, evidence_by_id) for claim in claims]

        total_citations = sum(len(claim.citation_ids) for claim in claims)
        valid_citations = sum(len(check.valid_citation_ids) for check in checks)
        citation_coverage = valid_citations / total_citations if total_citations else 0.0
        claim_coverage = sum(check.coverage for check in checks) / len(checks) if checks else 0.0
        counts = {
            verdict: sum(check.verdict is verdict for check in checks)
            for verdict in EvidenceVerdict
        }
        warnings = [
            "Deterministic lexical verification is not semantic entailment; matched text requires review.",
            *normalization_warnings,
            *[warning for check in checks for warning in check.warnings],
        ]
        return EvidenceVerificationReport(
            claim_checks=checks,
            citation_coverage=round(citation_coverage, 4),
            claim_coverage=round(claim_coverage, 4),
            supported_claim_count=counts[EvidenceVerdict.SUPPORTED],
            partial_claim_count=counts[EvidenceVerdict.PARTIAL],
            unsupported_claim_count=counts[EvidenceVerdict.UNSUPPORTED],
            contradicted_claim_count=counts[EvidenceVerdict.CONTRADICTED],
            warnings=list(dict.fromkeys(warnings)),
        )


def verify_evidence(
    report: ReportLike,
    evidence: EvidenceInput,
) -> EvidenceVerificationReport:
    """Convenience wrapper for the default deterministic verifier."""

    return DeterministicLexicalVerifier().verify(report, evidence)


# Short aliases make the contracts easy to discover without hiding their intent.
VerificationReport = EvidenceVerificationReport
ClaimCheck = ClaimEvidenceCheck
EvidenceMatch = MatchedEvidence


__all__ = [
    "ClaimCheck",
    "ClaimEvidenceCheck",
    "ClaimReference",
    "DeterministicLexicalVerifier",
    "EvidenceContractError",
    "EvidenceInput",
    "EvidenceMatch",
    "EvidenceRecord",
    "EvidenceVerifier",
    "EvidenceVerdict",
    "EvidenceVerificationReport",
    "MatchedEvidence",
    "VerificationReport",
    "extract_claim_references",
    "verify_evidence",
]
