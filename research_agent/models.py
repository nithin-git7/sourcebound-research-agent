"""Strict data contracts, provenance checks, and report evidence metrics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr

if TYPE_CHECKING:
    from .semantic_verification import SemanticVerificationReport
    from .verification import EvidenceVerificationReport


class StrictModel(BaseModel):
    """Closed Pydantic contract used at tool and report boundaries."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_assignment=True,
    )


class Source(StrictModel):
    """A source record returned by a provider and safe to cite by ID."""

    id: StrictStr = Field(min_length=2, max_length=120)
    provider: StrictStr = Field(min_length=2, max_length=80)
    title: StrictStr = Field(min_length=1, max_length=300)
    url: StrictStr = Field(min_length=1, max_length=2_000)
    snippet: StrictStr = Field(min_length=1, max_length=5_000)
    kind: StrictStr = Field(default="web", min_length=2, max_length=40)
    published_at: StrictStr | None = None
    authors: list[StrictStr] = Field(default_factory=list, max_length=12)
    credibility: Literal["high", "medium", "unknown"] = "unknown"
    # Optional provenance for sources returned by hosted search. ``snippet``
    # remains the backwards-compatible evidence field for older providers.
    evidence_text: StrictStr | None = Field(default=None, max_length=5_000)
    start_index: StrictInt | None = Field(default=None, ge=0)
    end_index: StrictInt | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProviderStatus(StrictModel):
    """Provider health in the source fan-out."""

    provider_id: StrictStr = Field(min_length=2, max_length=80)
    ok: StrictBool
    result_count: StrictInt = Field(default=0, ge=0)
    error: StrictStr | None = None

    @property
    def provider(self) -> str:
        return self.provider_id

    @property
    def status(self) -> Literal["ok", "error"]:
        return "ok" if self.ok else "error"

    @property
    def source_count(self) -> int:
        return self.result_count


class SearchBundle(StrictModel):
    """The structured result of one multi-provider search call."""

    query: StrictStr = Field(min_length=1)
    sources: list[Source]
    provider_statuses: list[ProviderStatus]

    @property
    def provider_status(self) -> list[ProviderStatus]:
        return self.provider_statuses


class Finding(StrictModel):
    finding_id: StrictStr = Field(min_length=1, max_length=40)
    statement: StrictStr = Field(min_length=1, max_length=1_000)
    importance: Literal["high", "medium", "low"]
    confidence: StrictFloat = Field(ge=0.0, le=1.0)
    citation_ids: list[StrictStr] = Field(min_length=1, max_length=12)


class SourceView(StrictModel):
    source_id: StrictStr = Field(min_length=2, max_length=120)
    position: StrictStr = Field(min_length=1, max_length=700)


class ComparisonPoint(StrictModel):
    dimension: StrictStr = Field(min_length=1, max_length=200)
    consensus: StrictStr = Field(min_length=1, max_length=800)
    disagreements: list[StrictStr] = Field(default_factory=list, max_length=8)
    source_views: list[SourceView] = Field(min_length=1, max_length=12)


class ResearchReportDraft(StrictModel):
    """The structured JSON object synthesized by the model."""

    executive_summary: StrictStr = Field(min_length=1, max_length=2_000)
    key_findings: list[Finding] = Field(min_length=1, max_length=8)
    comparison: list[ComparisonPoint] = Field(min_length=1, max_length=8)
    limitations: list[StrictStr] = Field(min_length=1, max_length=8)


class EvidenceAudit(StrictModel):
    """Deterministic quality metrics calculated after model synthesis."""

    citation_coverage: StrictFloat = Field(ge=0.0, le=1.0)
    grounding_score: StrictFloat = Field(ge=0.0, le=1.0)
    source_diversity: StrictFloat = Field(ge=0.0, le=1.0)
    comparison_quality: StrictFloat = Field(ge=0.0, le=1.0)
    score: StrictFloat = Field(ge=0.0, le=1.0)
    cited_finding_count: StrictInt = Field(ge=0)
    finding_count: StrictInt = Field(ge=0)
    provider_count: StrictInt = Field(ge=0)
    unresolved_citations: list[StrictStr] = Field(default_factory=list)
    warnings: list[StrictStr] = Field(default_factory=list)


class ToolCallTrace(StrictModel):
    name: StrictStr
    query: StrictStr
    source_count: StrictInt = Field(ge=0)
    providers: list[StrictStr] = Field(default_factory=list)
    planned_queries: list[StrictStr] = Field(default_factory=list, max_length=8)
    covered_intents: list[StrictStr] = Field(default_factory=list, max_length=8)
    missing_intents: list[StrictStr] = Field(default_factory=list, max_length=8)
    retrieval_coverage: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    stop_reason: StrictStr | None = None


class ResearchReport(StrictModel):
    """Final application-owned report envelope."""

    run_id: StrictStr
    question: StrictStr
    executive_summary: StrictStr
    key_findings: list[Finding]
    comparison: list[ComparisonPoint]
    limitations: list[StrictStr]
    sources: list[Source]
    provider_status: list[ProviderStatus]
    audit: EvidenceAudit
    # Added by the application after synthesis; the model only produces the
    # strict ResearchReportDraft contract above.
    verification: EvidenceVerificationReport | None = None
    semantic_verification: SemanticVerificationReport | None = None
    model: StrictStr
    tool_calls: list[ToolCallTrace]
    generated_at: StrictStr

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        question: str,
        draft: ResearchReportDraft,
        sources: list[Source],
        provider_status: list[ProviderStatus],
        audit: EvidenceAudit,
        verification: EvidenceVerificationReport | None = None,
        semantic_verification: SemanticVerificationReport | None = None,
        model: str,
        tool_calls: list[ToolCallTrace],
    ) -> "ResearchReport":
        return cls(
            run_id=run_id,
            question=question,
            executive_summary=draft.executive_summary,
            key_findings=draft.key_findings,
            comparison=draft.comparison,
            limitations=draft.limitations,
            sources=sources,
            provider_status=provider_status,
            audit=audit,
            verification=verification,
            semantic_verification=semantic_verification,
            model=model,
            tool_calls=tool_calls,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )


class GroundingValidationError(ValueError):
    """Raised when a draft cites evidence outside the search result set."""

    def __init__(self, unresolved: list[str]):
        self.unresolved = sorted(set(unresolved))
        super().__init__("unknown citation IDs: " + ", ".join(self.unresolved))


GroundingError = GroundingValidationError
ReportDraft = ResearchReportDraft


# Compatibility models kept for callers that want a minimal claim/section shape.
class SourceCitation(StrictModel):
    id: StrictStr = Field(min_length=1)
    title: StrictStr = Field(min_length=1)
    url: StrictStr = Field(min_length=1)
    snippet: StrictStr = Field(min_length=1)
    publisher: StrictStr | None = None
    published_at: StrictStr | None = None


class ReportClaim(StrictModel):
    statement: StrictStr = Field(min_length=1)
    citation_ids: list[StrictStr] = Field(min_length=1)


class ReportSection(StrictModel):
    heading: StrictStr = Field(min_length=1)
    claims: list[ReportClaim] = Field(min_length=1)


Citation = SourceCitation
Claim = ReportClaim
Section = ReportSection


def _evidence_ids(evidence: Mapping[str, Any] | Iterable[Any]) -> set[str]:
    if isinstance(evidence, Mapping):
        return {str(key) for key in evidence}
    identifiers: set[str] = set()
    for item in evidence:
        if isinstance(item, Mapping):
            value = item.get("id", item.get("citation_id", item.get("source_id")))
        else:
            value = getattr(item, "id", None)
            if value is None:
                value = getattr(item, "citation_id", getattr(item, "source_id", None))
        if value is not None:
            identifiers.add(str(value))
    return identifiers


def referenced_source_ids(draft: ResearchReportDraft) -> set[str]:
    references = {
        citation_id
        for finding in draft.key_findings
        for citation_id in finding.citation_ids
    }
    references.update(
        view.source_id
        for comparison in draft.comparison
        for view in comparison.source_views
    )
    return references


def _source_items(evidence: Mapping[str, Any] | Iterable[Any]) -> list[Any]:
    return list(evidence.values()) if isinstance(evidence, Mapping) else list(evidence)


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def audit_evidence(
    draft: ResearchReportDraft,
    evidence: Mapping[str, Any] | Iterable[Any],
) -> EvidenceAudit:
    """Compute a grounding audit for a model draft and known evidence IDs."""

    known = _evidence_ids(evidence)
    evidence_items = _source_items(evidence)
    references = referenced_source_ids(draft)
    unresolved = sorted(references - known)
    finding_references = [finding.citation_ids for finding in draft.key_findings]
    cited_count = sum(
        bool(references_for_finding)
        and all(source_id in known for source_id in references_for_finding)
        for references_for_finding in finding_references
    )
    provider_names = {_value(source, "provider") for source in evidence_items}
    provider_names.discard(None)
    source_diversity = min(1.0, len(provider_names) / 2.0)
    comparison_scores: list[float] = []
    for point in draft.comparison:
        point_ids = {view.source_id for view in point.source_views}
        point_providers = {
            _value(source, "provider")
            for source in evidence_items
            if _value(source, "id") in point_ids
        }
        point_providers.discard(None)
        comparison_scores.append(min(1.0, len(point_providers) / 2.0))
    comparison_quality = sum(comparison_scores) / max(1, len(comparison_scores))
    citation_coverage = cited_count / max(1, len(draft.key_findings))
    grounding_score = 1.0 if not unresolved else 0.0
    warnings: list[str] = []
    if len(provider_names) < 2:
        warnings.append("Fewer than two independent providers returned usable sources.")
    if unresolved:
        warnings.append("At least one report citation could not be resolved.")
    score = (
        citation_coverage * 0.35
        + grounding_score * 0.25
        + source_diversity * 0.20
        + comparison_quality * 0.20
    )
    return EvidenceAudit(
        citation_coverage=round(citation_coverage, 4),
        grounding_score=round(grounding_score, 4),
        source_diversity=round(source_diversity, 4),
        comparison_quality=round(comparison_quality, 4),
        score=round(score, 4),
        cited_finding_count=cited_count,
        finding_count=len(draft.key_findings),
        provider_count=len(provider_names),
        unresolved_citations=unresolved,
        warnings=warnings,
    )


def validate_grounding(
    draft: ResearchReportDraft,
    evidence: Mapping[str, Any] | Iterable[Any],
) -> EvidenceAudit:
    audit = audit_evidence(draft, evidence)
    if audit.unresolved_citations:
        raise GroundingValidationError(audit.unresolved_citations)
    return audit


def build_audit(draft: ResearchReportDraft, sources: list[Source]) -> EvidenceAudit:
    return audit_evidence(draft, {source.id: source for source in sources})


# ``verification.py`` imports the report models to remain provider-agnostic. It
# is therefore resolved after this module has defined ResearchReport, avoiding a
# runtime import cycle while keeping the final envelope fully typed.
from .verification import EvidenceVerificationReport as _EvidenceVerificationReport
from .semantic_verification import SemanticVerificationReport as _SemanticVerificationReport

ResearchReport.model_rebuild(
    _types_namespace={
        "EvidenceVerificationReport": _EvidenceVerificationReport,
        "SemanticVerificationReport": _SemanticVerificationReport,
    }
)
