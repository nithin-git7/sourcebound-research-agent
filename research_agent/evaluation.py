from __future__ import annotations

import json
import re
from time import perf_counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .models import (
    ComparisonPoint,
    Finding,
    ResearchReport,
    ResearchReportDraft,
    Source,
    SourceView,
    build_audit,
)
from .offline import build_offline_agent
from .verification import EvidenceVerdict, verify_evidence


BENCHMARK_DATA_PATH = Path(__file__).resolve().parent / "data" / "research_benchmark.json"
DEFAULT_BENCHMARK_ID = "sourcebound-research-benchmark-v2"
DEFAULT_BENCHMARK_VERSION = "2.0.0"
FIXTURE_METRIC_DISCLOSURE = (
    "Recorded-fixture metrics are deterministic lexical and structural proxy metrics over "
    "curated source excerpts; they are not live-web quality measurements, semantic "
    "entailment judgments, medical advice, or a substitute for expert human review."
)


class ExpectedConcept(BaseModel):
    """A concept that should be retrievable and reflected in the final answer."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=80)
    label: str = Field(min_length=1, max_length=160)
    terms: list[str] = Field(min_length=1, max_length=12)
    weight: float = Field(default=1.0, gt=0.0, le=10.0)


class BenchmarkCriterion(BaseModel):
    """A machine-checkable threshold attached to one benchmark case."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=300)
    metric: Literal["retrieval_recall", "claim_support", "completeness"]
    minimum: float = Field(ge=0.0, le=1.0)


class RecordedSource(BaseModel):
    """A curated, attributed excerpt used only for deterministic evaluation."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=120)
    provider: str = Field(min_length=1, max_length=120)
    title: str = Field(min_length=1, max_length=300)
    url: str = Field(min_length=1, max_length=2_000)
    text: str = Field(min_length=1, max_length=5_000)


class RecordedFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statement: str = Field(min_length=1, max_length=1_000)
    citation_ids: list[str] = Field(min_length=1, max_length=12)


class RecordedFixture(BaseModel):
    """A frozen research artifact; it does not claim to be a live agent response."""

    model_config = ConfigDict(extra="forbid")

    sources: list[RecordedSource] = Field(min_length=2, max_length=8)
    executive_summary: str = Field(min_length=1, max_length=2_000)
    findings: list[RecordedFinding] = Field(min_length=1, max_length=8)
    comparison_source_ids: list[str] = Field(min_length=2, max_length=8)


class RegressionThresholds(BaseModel):
    """Aggregate quality floor for detecting benchmark regressions in CI."""

    model_config = ConfigDict(extra="forbid")

    minimum_pass_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    minimum_mean_score: float = Field(default=0.85, ge=0.0, le=1.0)
    minimum_mean_retrieval_recall: float = Field(default=0.80, ge=0.0, le=1.0)
    minimum_mean_claim_support: float = Field(default=0.80, ge=0.0, le=1.0)
    minimum_mean_completeness: float = Field(default=0.80, ge=0.0, le=1.0)


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    question: str
    minimum_score: float = Field(default=0.8, ge=0.0, le=1.0)
    minimum_providers: int = Field(default=2, ge=1, le=10)
    category: str = "general"
    benchmark_id: str = DEFAULT_BENCHMARK_ID
    benchmark_version: str = DEFAULT_BENCHMARK_VERSION
    expected_concepts: list[ExpectedConcept] = Field(default_factory=list)
    criteria: list[BenchmarkCriterion] = Field(default_factory=list)
    fixture: RecordedFixture | None = None


class BenchmarkDefinition(BaseModel):
    """Versioned, data-driven benchmark metadata and cases."""

    model_config = ConfigDict(extra="forbid")

    benchmark_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    name: str = Field(min_length=1)
    mode: Literal["fixture"] = "fixture"
    metric_disclosure: str = Field(min_length=1)
    regression_thresholds: RegressionThresholds = Field(default_factory=RegressionThresholds)
    cases: list[EvaluationCase] = Field(min_length=1)


class EvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    score: float = Field(ge=0.0, le=1.0)
    citation_coverage: float = Field(ge=0.0, le=1.0)
    grounding: float = Field(ge=0.0, le=1.0)
    source_diversity: float = Field(ge=0.0, le=1.0)
    comparison_quality: float = Field(ge=0.0, le=1.0)
    provider_count: int = Field(ge=0)
    passed: bool
    reasons: list[str] = Field(default_factory=list)
    benchmark_id: str = DEFAULT_BENCHMARK_ID
    benchmark_version: str = DEFAULT_BENCHMARK_VERSION
    category: str = "general"
    retrieval_recall: float = Field(default=1.0, ge=0.0, le=1.0)
    claim_support: float = Field(default=1.0, ge=0.0, le=1.0)
    completeness: float = Field(default=1.0, ge=0.0, le=1.0)
    expected_concept_count: int = Field(default=0, ge=0)
    retrieved_concept_count: int = Field(default=0, ge=0)
    covered_concept_count: int = Field(default=0, ge=0)
    claim_count: int = Field(default=0, ge=0)
    supported_claim_count: int = Field(default=0, ge=0)
    retrieval_missing_concepts: list[str] = Field(default_factory=list)
    answer_missing_concepts: list[str] = Field(default_factory=list)
    evaluation_duration_ms: float = Field(default=0.0, ge=0.0)


class EvaluationSuite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[EvaluationResult]
    mean_score: float = Field(ge=0.0, le=1.0)
    passed: bool
    benchmark_id: str = DEFAULT_BENCHMARK_ID
    benchmark_version: str = DEFAULT_BENCHMARK_VERSION
    benchmark_name: str = "Sourcebound recorded-fixture research benchmark"
    metric_disclosure: str = FIXTURE_METRIC_DISCLOSURE
    case_count: int = Field(default=0, ge=0)
    pass_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    mean_retrieval_recall: float = Field(default=0.0, ge=0.0, le=1.0)
    mean_claim_support: float = Field(default=0.0, ge=0.0, le=1.0)
    mean_completeness: float = Field(default=0.0, ge=0.0, le=1.0)
    total_evaluation_duration_ms: float = Field(default=0.0, ge=0.0)
    regression_thresholds: RegressionThresholds = Field(default_factory=RegressionThresholds)
    regression_passed: bool = False
    regression_failures: list[str] = Field(default_factory=list)


def load_benchmark(path: str | Path | None = None) -> BenchmarkDefinition:
    """Load and validate a versioned benchmark definition from JSON."""

    benchmark_path = Path(path) if path is not None else BENCHMARK_DATA_PATH
    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    return BenchmarkDefinition.model_validate(payload)


DEFAULT_BENCHMARK = load_benchmark()
DEFAULT_CASES = DEFAULT_BENCHMARK.cases


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "answer", "are", "as", "at", "be", "by", "can", "for",
    "from", "helps", "in", "is", "it", "model", "of", "on", "or", "part", "the",
    "their", "this", "to", "with",
}


def _normalized(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.lower()))


def _concept_hit(concept: ExpectedConcept, text: str) -> bool:
    normalized_text = _normalized(text)
    return any(_normalized(term) in normalized_text for term in concept.terms)


def _concept_metrics(
    concepts: list[ExpectedConcept],
    text: str,
) -> tuple[float, int, list[str]]:
    if not concepts:
        return 1.0, 0, []
    total_weight = sum(concept.weight for concept in concepts)
    hits = [concept for concept in concepts if _concept_hit(concept, text)]
    score = sum(concept.weight for concept in hits) / total_weight
    missing = [concept.label for concept in concepts if concept not in hits]
    return round(score, 4), len(hits), missing


def _source_text(source: Any) -> str:
    if isinstance(source, dict):
        values = source.values()
    else:
        values = (
            getattr(source, "title", ""),
            getattr(source, "snippet", ""),
            getattr(source, "kind", ""),
            getattr(source, "provider", ""),
        )
    return " ".join(str(value) for value in values if value is not None)


def _report_text(report: ResearchReport) -> str:
    parts = [report.executive_summary]
    for finding in report.key_findings:
        parts.append(finding.statement)
    for comparison in report.comparison:
        parts.extend([comparison.dimension, comparison.consensus, *comparison.disagreements])
        parts.extend(view.position for view in comparison.source_views)
    parts.extend(report.limitations)
    return " ".join(parts)


def _claim_support_metrics(report: ResearchReport) -> tuple[float, int, int]:
    """Use the shared verifier for finding-level support metrics.

    Partial lexical matches receive half credit. This remains a transparent
    offline proxy for semantic entailment, not a semantic judge.
    """

    verification = verify_evidence(report, report.sources)
    finding_ids = {finding.finding_id for finding in report.key_findings}
    finding_checks = [
        check for check in verification.claim_checks if check.claim_id in finding_ids
    ]
    weighted_support = 0.0
    supported = 0
    for check in finding_checks:
        if check.verdict is EvidenceVerdict.SUPPORTED:
            weighted_support += 1.0
            supported += 1
        elif check.verdict is EvidenceVerdict.PARTIAL:
            weighted_support += 0.5
    claim_count = len(report.key_findings)
    return round(weighted_support / max(1, claim_count), 4), supported, claim_count


def _report_from_fixture(case: EvaluationCase) -> ResearchReport:
    """Convert one frozen benchmark artifact into the normal report contract."""

    fixture = case.fixture
    if fixture is None:
        raise ValueError(f"benchmark case {case.name!r} has no recorded fixture")

    sources = [
        Source(
            id=source.id,
            provider=source.provider,
            title=source.title,
            url=source.url,
            snippet=source.text,
            kind="recorded_fixture",
            credibility="unknown",
        )
        for source in fixture.sources
    ]
    source_by_id = {source.id: source for source in sources}
    unknown_comparison_ids = sorted(set(fixture.comparison_source_ids) - set(source_by_id))
    if unknown_comparison_ids:
        raise ValueError(
            f"benchmark case {case.name!r} has unknown comparison source IDs: "
            + ", ".join(unknown_comparison_ids)
        )

    findings = [
        Finding(
            finding_id=f"finding-{index}",
            statement=finding.statement,
            importance="high" if index == 1 else "medium",
            confidence=1.0,
            citation_ids=finding.citation_ids,
        )
        for index, finding in enumerate(fixture.findings, start=1)
    ]
    comparison = ComparisonPoint(
        dimension="Recorded source comparison",
        consensus=fixture.executive_summary,
        disagreements=[],
        source_views=[
            SourceView(
                source_id=source_id,
                position=source_by_id[source_id].snippet,
            )
            for source_id in fixture.comparison_source_ids
        ],
    )
    draft = ResearchReportDraft(
        executive_summary=fixture.executive_summary,
        key_findings=findings,
        comparison=[comparison],
        limitations=[
            "This is a curated recorded fixture, not a live-web research response or expert review."
        ],
    )
    audit = build_audit(draft, sources)
    return ResearchReport.create(
        run_id=f"benchmark-{case.name}",
        question=case.question,
        draft=draft,
        sources=sources,
        provider_status=[],
        audit=audit,
        model="recorded-fixture-v2",
        tool_calls=[],
    )


def evaluate_regression_thresholds(
    results: list[EvaluationResult],
    thresholds: RegressionThresholds,
) -> list[str]:
    """Return explicit aggregate threshold failures for CI and release reports."""

    count = len(results)
    aggregates = {
        "pass_rate": sum(result.passed for result in results) / max(1, count),
        "mean_score": sum(result.score for result in results) / max(1, count),
        "mean_retrieval_recall": sum(result.retrieval_recall for result in results)
        / max(1, count),
        "mean_claim_support": sum(result.claim_support for result in results) / max(1, count),
        "mean_completeness": sum(result.completeness for result in results) / max(1, count),
    }
    minimums = {
        "pass_rate": thresholds.minimum_pass_rate,
        "mean_score": thresholds.minimum_mean_score,
        "mean_retrieval_recall": thresholds.minimum_mean_retrieval_recall,
        "mean_claim_support": thresholds.minimum_mean_claim_support,
        "mean_completeness": thresholds.minimum_mean_completeness,
    }
    return [
        f"{metric} regression: {aggregates[metric]:.4f} < {minimum:.4f}"
        for metric, minimum in minimums.items()
        if aggregates[metric] < minimum
    ]


def evaluate_report(report: ResearchReport, case: EvaluationCase) -> EvaluationResult:
    """Score structural audit plus benchmark-specific deterministic proxy metrics."""

    audit = report.audit
    source_text = " ".join(_source_text(source) for source in report.sources)
    retrieval_recall, retrieved_count, retrieval_missing = _concept_metrics(
        case.expected_concepts, source_text
    )
    completeness, covered_count, answer_missing = _concept_metrics(
        case.expected_concepts, _report_text(report)
    )
    claim_support, supported_claim_count, claim_count = _claim_support_metrics(report)

    reasons: list[str] = []
    if audit.citation_coverage < 1.0:
        reasons.append("not every finding has fully resolved citations")
    if audit.grounding_score < 1.0:
        reasons.append("at least one citation is unresolved")
    if audit.provider_count < case.minimum_providers:
        reasons.append(f"fewer than {case.minimum_providers} providers returned sources")
    if audit.comparison_quality < 1.0:
        reasons.append("comparison points do not span two provider perspectives")
    if retrieval_missing:
        reasons.append("retrieval missed expected concepts: " + ", ".join(retrieval_missing))
    if answer_missing:
        reasons.append("answer omitted expected concepts: " + ", ".join(answer_missing))
    if claim_count and supported_claim_count < claim_count:
        reasons.append("not every finding meets the lexical claim-support proxy")

    metric_values = {
        "retrieval_recall": retrieval_recall,
        "claim_support": claim_support,
        "completeness": completeness,
    }
    for criterion in case.criteria:
        value = metric_values[criterion.metric]
        if value < criterion.minimum:
            reasons.append(
                f"criterion {criterion.id} failed: {value:.2f} < {criterion.minimum:.2f}"
            )

    score = (
        audit.score * 0.40
        + retrieval_recall * 0.20
        + claim_support * 0.20
        + completeness * 0.20
    )
    return EvaluationResult(
        name=case.name,
        score=round(score, 4),
        citation_coverage=audit.citation_coverage,
        grounding=audit.grounding_score,
        source_diversity=audit.source_diversity,
        comparison_quality=audit.comparison_quality,
        provider_count=audit.provider_count,
        passed=score >= case.minimum_score and not reasons,
        reasons=reasons,
        benchmark_id=case.benchmark_id,
        benchmark_version=case.benchmark_version,
        category=case.category,
        retrieval_recall=retrieval_recall,
        claim_support=claim_support,
        completeness=completeness,
        expected_concept_count=len(case.expected_concepts),
        retrieved_concept_count=retrieved_count,
        covered_concept_count=covered_count,
        claim_count=claim_count,
        supported_claim_count=supported_claim_count,
        retrieval_missing_concepts=retrieval_missing,
        answer_missing_concepts=answer_missing,
    )


def run_evaluation_suite(
    cases: list[EvaluationCase] | None = None,
    *,
    agent_factory: Any = build_offline_agent,
) -> EvaluationSuite:
    """Run the versioned offline benchmark while preserving the old entry point."""

    selected_cases = list(DEFAULT_CASES if cases is None else cases)
    results: list[EvaluationResult] = []
    for case in selected_cases:
        started = perf_counter()
        report = _report_from_fixture(case) if case.fixture else agent_factory().run(case.question)
        result = evaluate_report(report, case)
        results.append(
            result.model_copy(
                update={"evaluation_duration_ms": round((perf_counter() - started) * 1_000, 3)}
            )
        )

    case_count = len(results)
    mean_score = sum(result.score for result in results) / max(1, case_count)
    pass_rate = sum(result.passed for result in results) / max(1, case_count)
    mean_retrieval = sum(result.retrieval_recall for result in results) / max(1, case_count)
    mean_claim_support = sum(result.claim_support for result in results) / max(1, case_count)
    mean_completeness = sum(result.completeness for result in results) / max(1, case_count)
    total_duration = sum(result.evaluation_duration_ms for result in results)

    if selected_cases and all(
        case.benchmark_id == DEFAULT_BENCHMARK.benchmark_id for case in selected_cases
    ):
        benchmark_id = DEFAULT_BENCHMARK.benchmark_id
        benchmark_version = DEFAULT_BENCHMARK.version
        benchmark_name = DEFAULT_BENCHMARK.name
        disclosure = DEFAULT_BENCHMARK.metric_disclosure
        thresholds = DEFAULT_BENCHMARK.regression_thresholds
    else:
        benchmark_id = "custom"
        benchmark_version = "custom"
        benchmark_name = "Custom evaluation cases"
        disclosure = FIXTURE_METRIC_DISCLOSURE
        thresholds = RegressionThresholds()

    regression_failures = evaluate_regression_thresholds(results, thresholds)

    return EvaluationSuite(
        results=results,
        mean_score=round(mean_score, 4),
        passed=all(result.passed for result in results) and not regression_failures,
        benchmark_id=benchmark_id,
        benchmark_version=benchmark_version,
        benchmark_name=benchmark_name,
        metric_disclosure=disclosure,
        case_count=case_count,
        pass_rate=round(pass_rate, 4),
        mean_retrieval_recall=round(mean_retrieval, 4),
        mean_claim_support=round(mean_claim_support, 4),
        mean_completeness=round(mean_completeness, 4),
        total_evaluation_duration_ms=round(total_duration, 3),
        regression_thresholds=thresholds,
        regression_passed=not regression_failures,
        regression_failures=regression_failures,
    )
