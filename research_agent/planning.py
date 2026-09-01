"""Bounded, provider-agnostic retrieval planning and ranking.

The planner deliberately has no network or provider dependencies.  A future
``ResearchAgent`` integration can create a plan, execute each
``SearchIntent.query`` through its existing search tool, and pass a mapping of
``{query: [Source, ...]}`` to :meth:`RetrievalPlanner.rank`.

Example::

    plan = RetrievalPlanner().create_plan(
        question,
        required_source_kinds=["academic", "official"],
        freshness_hint="recent",
    )
    bundles = {
        intent.query: search_tool.search(
            intent.query,
            max_results=plan.per_query_result_limit,
        ).sources
        for intent in plan.intents
    }
    result = RetrievalPlanner().rank(plan, bundles)
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, timezone
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit
import re

from pydantic import ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr

from .models import Source, StrictModel


FreshnessMode = Literal["any", "recent", "current", "historical"]
StopReason = Literal[
    "coverage_met",
    "budget_exhausted",
    "query_cap_reached",
    "needs_more_evidence",
]


class FreshnessHint(StrictModel):
    """A deterministic freshness preference used by the ranking signal."""

    mode: FreshnessMode = "any"
    max_age_days: StrictInt | None = Field(default=None, gt=0, le=36_500)
    reference_date: StrictStr | None = None


class SearchIntent(StrictModel):
    """One focused retrieval objective and the query used to pursue it."""

    intent_id: StrictStr = Field(min_length=2, max_length=40)
    label: StrictStr = Field(min_length=2, max_length=80)
    purpose: StrictStr = Field(min_length=1, max_length=240)
    query: StrictStr = Field(min_length=3, max_length=500)
    priority: StrictInt = Field(ge=1, le=10)


class StopCriteria(StrictModel):
    """Conditions that tell an executor when more retrieval is unnecessary."""

    target_coverage: StrictFloat = Field(default=0.8, ge=0.0, le=1.0)
    min_unique_sources: StrictInt = Field(default=3, ge=1, le=500)
    min_provider_count: StrictInt = Field(default=2, ge=0, le=100)
    require_all_source_kinds: StrictBool = True


class ResearchPlan(StrictModel):
    """Closed plan contract consumed by a retrieval executor."""

    question: StrictStr = Field(min_length=3, max_length=1_000)
    intents: list[SearchIntent] = Field(min_length=1, max_length=8)
    required_source_kinds: list[StrictStr] = Field(default_factory=list, max_length=12)
    freshness: FreshnessHint = Field(default_factory=FreshnessHint)
    max_query_count: StrictInt = Field(ge=1, le=8)
    per_query_result_limit: StrictInt = Field(ge=1, le=50)
    overall_budget: StrictInt = Field(ge=1, le=400)
    stop: StopCriteria = Field(default_factory=StopCriteria)


class SourceHit(StrictModel):
    """A source plus the intent query that returned it."""

    query: StrictStr = Field(min_length=1, max_length=500)
    source: Source


class RankedSource(StrictModel):
    """A source with explainable component scores and selection metadata."""

    rank: StrictInt = Field(ge=1)
    source: Source
    score: StrictFloat = Field(ge=0.0, le=1.0)
    relevance_score: StrictFloat = Field(ge=0.0, le=1.0)
    diversity_score: StrictFloat = Field(ge=0.0, le=1.0)
    credibility_score: StrictFloat = Field(ge=0.0, le=1.0)
    recency_score: StrictFloat = Field(ge=0.0, le=1.0)
    matched_intents: list[StrictStr] = Field(default_factory=list, max_length=8)


class StopDecision(StrictModel):
    """A typed explanation of whether the executor should stop."""

    stop: StrictBool
    reason: StopReason
    coverage: StrictFloat = Field(ge=0.0, le=1.0)
    intent_coverage: StrictFloat = Field(ge=0.0, le=1.0)
    source_kind_coverage: StrictFloat = Field(ge=0.0, le=1.0)
    provider_count: StrictInt = Field(ge=0)
    unique_source_count: StrictInt = Field(ge=0)
    queries_attempted: StrictInt = Field(ge=0)
    budget_used: StrictInt = Field(ge=0)
    covered_intents: list[StrictStr] = Field(default_factory=list, max_length=8)
    missing_intents: list[StrictStr] = Field(default_factory=list, max_length=8)
    missing_source_kinds: list[StrictStr] = Field(default_factory=list, max_length=12)


class RetrievalResult(StrictModel):
    """Bounded ranked evidence and the stop decision for one plan."""

    plan: ResearchPlan
    sources: list[RankedSource]
    raw_result_count: StrictInt = Field(ge=0)
    deduplicated_count: StrictInt = Field(ge=0)
    queries_considered: StrictInt = Field(ge=0)
    budget_used: StrictInt = Field(ge=0)
    provider_count: StrictInt = Field(ge=0)
    coverage: StrictFloat = Field(ge=0.0, le=1.0)
    covered_intents: list[StrictStr] = Field(default_factory=list, max_length=8)
    missing_intents: list[StrictStr] = Field(default_factory=list, max_length=8)
    covered_source_kinds: list[StrictStr] = Field(default_factory=list, max_length=12)
    missing_source_kinds: list[StrictStr] = Field(default_factory=list, max_length=12)
    stop: StopDecision


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "for",
    "from", "how", "in", "is", "it", "of", "on", "or", "that", "the",
    "this", "to", "what", "when", "where", "which", "with", "why",
}
_CREDIBILITY = {"high": 1.0, "medium": 0.65, "unknown": 0.35}


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(str(text).casefold())
        if token not in _STOPWORDS and len(token) > 1
    }


def _clean_query(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip()).strip(" ?")


def _normalise_kind(kind: str) -> str:
    return re.sub(r"\s+", "-", str(kind).strip().casefold())


def _canonical_url(url: str) -> str:
    parsed = urlsplit(str(url).strip())
    if not parsed.netloc:
        return str(url).strip().casefold()
    host = (parsed.hostname or "").casefold()
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit((parsed.scheme.casefold(), host, path, parsed.query, ""))


def _parse_date(value: Any) -> date | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if re.fullmatch(r"\d{4}", text):
            return date(int(text), 7, 1)
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except (TypeError, ValueError):
        try:
            return date.fromisoformat(text[:10])
        except (TypeError, ValueError):
            return None


def _reference_date(hint: FreshnessHint) -> date:
    parsed = _parse_date(hint.reference_date)
    return parsed or datetime.now(timezone.utc).date()


def _recency_score(source: Source, hint: FreshnessHint) -> float:
    published = _parse_date(source.published_at)
    if published is None:
        return 0.25 if hint.mode == "any" else 0.0
    age_days = max(0, (_reference_date(hint) - published).days)
    if hint.mode == "historical":
        horizon = hint.max_age_days or 3_650
        return round(min(1.0, age_days / horizon), 4)
    if hint.mode in {"recent", "current"}:
        horizon = hint.max_age_days or (90 if hint.mode == "current" else 365)
        return round(max(0.0, 1.0 - (age_days / horizon)), 4)
    return round(max(0.0, 1.0 - (age_days / 3_650)), 4)


def _relevance_score(query: str, source: Source) -> float:
    query_terms = _tokens(query)
    if not query_terms:
        return 0.0
    title_terms = _tokens(source.title)
    snippet_terms = _tokens(source.snippet)
    title_coverage = len(query_terms & title_terms) / len(query_terms)
    snippet_coverage = len(query_terms & snippet_terms) / len(query_terms)
    return round(min(1.0, title_coverage * 0.7 + snippet_coverage * 0.3), 4)


def _source_score(
    query: str,
    source: Source,
    hint: FreshnessHint,
) -> tuple[float, float, float, float]:
    relevance = _relevance_score(query, source)
    credibility = _CREDIBILITY.get(source.credibility, 0.35)
    recency = _recency_score(source, hint)
    base = relevance * 0.55 + credibility * 0.15 + recency * 0.10
    return relevance, credibility, recency, round(base, 4)


def _coerce_freshness(value: FreshnessHint | FreshnessMode | Mapping[str, Any] | None) -> FreshnessHint:
    if value is None:
        return FreshnessHint()
    if isinstance(value, FreshnessHint):
        return value
    if isinstance(value, str):
        return FreshnessHint(mode=value)  # type: ignore[arg-type]
    return FreshnessHint.model_validate(value)


def _normalise_kinds(values: Sequence[str] | None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values or ():
        kind = _normalise_kind(value)
        if kind and kind not in seen:
            result.append(kind)
            seen.add(kind)
    return result


def decompose_question(question: str, *, max_query_count: int = 4, freshness_hint: FreshnessHint | FreshnessMode | Mapping[str, Any] | None = None) -> list[SearchIntent]:
    """Create a small deterministic set of focused intents without network calls."""

    clean = _clean_query(question)
    if len(clean) < 3:
        raise ValueError("question must contain at least three non-whitespace characters")
    if not 1 <= int(max_query_count) <= 8:
        raise ValueError("max_query_count must be between 1 and 8")
    hint = _coerce_freshness(freshness_hint)
    suffix = {
        "any": "",
        "recent": " recent evidence",
        "current": " current developments",
        "historical": " historical context",
    }[hint.mode]
    lowered = clean.casefold()
    intents: list[tuple[str, str, str, str, int]] = [
        ("overview", "Overview", "establish the core concepts and scope", f"{clean}{suffix}", 10),
        ("evidence", "Evidence", "find empirical, technical, or scholarly evidence", f"{clean} evidence findings studies{suffix}", 9),
        ("tradeoffs", "Trade-offs", "surface benefits, limitations, risks, and trade-offs", f"{clean} benefits limitations risks trade-offs{suffix}", 8),
        ("comparison", "Comparison", "compare approaches, alternatives, or source perspectives", f"{clean} comparison alternatives perspectives{suffix}", 7),
    ]
    if any(term in lowered for term in ("compare", "versus", " vs ", "alternative")):
        intents.insert(1, intents.pop(3))
    if any(term in lowered for term in ("latest", "current", "today", "recent")) and hint.mode == "any":
        intents[0] = (intents[0][0], intents[0][1], intents[0][2], f"{clean} latest current developments", intents[0][4])
    selected = intents[: int(max_query_count)]
    return [
        SearchIntent(
            intent_id=item[0],
            label=item[1],
            purpose=item[2],
            query=_clean_query(item[3]),
            priority=item[4],
        )
        for item in selected
    ]


def build_research_plan(
    question: str,
    *,
    required_source_kinds: Sequence[str] | None = None,
    freshness_hint: FreshnessHint | FreshnessMode | Mapping[str, Any] | None = None,
    max_query_count: int = 4,
    per_query_result_limit: int = 5,
    overall_budget: int = 20,
    stop: StopCriteria | Mapping[str, Any] | None = None,
) -> ResearchPlan:
    """Build a bounded plan with normalized requirements and explicit stops."""

    if not 1 <= int(per_query_result_limit) <= 50:
        raise ValueError("per_query_result_limit must be between 1 and 50")
    if not 1 <= int(overall_budget) <= 400:
        raise ValueError("overall_budget must be between 1 and 400")
    freshness = _coerce_freshness(freshness_hint)
    stop_contract = stop if isinstance(stop, StopCriteria) else StopCriteria.model_validate(stop or {})
    return ResearchPlan(
        question=_clean_query(question),
        intents=decompose_question(
            question,
            max_query_count=max_query_count,
            freshness_hint=freshness,
        ),
        required_source_kinds=_normalise_kinds(required_source_kinds),
        freshness=freshness,
        max_query_count=int(max_query_count),
        per_query_result_limit=int(per_query_result_limit),
        overall_budget=int(overall_budget),
        stop=stop_contract,
    )


def _as_hits(
    results: Mapping[str, Iterable[Source]] | Iterable[SourceHit | Source],
    plan: ResearchPlan,
) -> tuple[list[SourceHit], int]:
    """Admit only bounded, typed hits and return (hits, queries_considered)."""

    grouped: list[tuple[str, list[Source]]] = []
    if isinstance(results, Mapping):
        for query, values in results.items():
            grouped.append((str(query), list(values)))
    else:
        for item in results:
            if isinstance(item, SourceHit):
                query, source = item.query, item.source
            elif isinstance(item, Source):
                query, source = plan.question, item
            else:
                raise TypeError("results must contain Source or SourceHit records")
            existing = next((group for group in grouped if group[0] == query), None)
            if existing is None:
                grouped.append((query, [source]))
            else:
                existing[1].append(source)

    allowed_queries = {intent.query for intent in plan.intents[: plan.max_query_count]}
    accepted: list[SourceHit] = []
    considered_queries = 0
    for query, values in grouped:
        if allowed_queries and query not in allowed_queries and query != plan.question:
            continue
        if considered_queries >= plan.max_query_count:
            break
        considered_queries += 1
        for source in values[: plan.per_query_result_limit]:
            if len(accepted) >= plan.overall_budget:
                return accepted, considered_queries
            accepted.append(SourceHit(query=query, source=source))
    return accepted, considered_queries


def _intent_matches(plan: ResearchPlan, query: str, source: Source) -> list[str]:
    query_intent = next((intent for intent in plan.intents if intent.query == query), None)
    if query_intent is not None:
        return [query_intent.intent_id]
    matches = [
        intent.intent_id
        for intent in plan.intents
        if _relevance_score(intent.query, source) >= 0.2
    ]
    return matches


def _coverage(
    plan: ResearchPlan,
    sources: Sequence[RankedSource],
    queries_attempted: int,
    budget_used: int,
) -> StopDecision:
    source_values = [item.source for item in sources]
    covered_intents = sorted({intent for item in sources for intent in item.matched_intents})
    intent_ids = [intent.intent_id for intent in plan.intents]
    missing_intents = [intent for intent in intent_ids if intent not in covered_intents]
    required = plan.required_source_kinds
    present_kinds = {_normalise_kind(source.kind) for source in source_values}
    covered_kinds = sorted(set(required) & present_kinds)
    missing_kinds = [kind for kind in required if kind not in present_kinds]
    intent_coverage = len(covered_intents) / max(1, len(intent_ids))
    kind_coverage = len(covered_kinds) / max(1, len(required)) if required else 1.0
    providers = {source.provider for source in source_values}
    provider_target = plan.stop.min_provider_count
    provider_coverage = min(1.0, len(providers) / provider_target) if provider_target else 1.0
    coverage = round(intent_coverage * 0.55 + kind_coverage * 0.25 + provider_coverage * 0.20, 4)
    requirements_met = (
        coverage >= plan.stop.target_coverage
        and len(sources) >= plan.stop.min_unique_sources
        and (not plan.stop.require_all_source_kinds or not missing_kinds)
        and len(providers) >= provider_target
    )
    if requirements_met:
        reason: StopReason = "coverage_met"
        should_stop = True
    elif budget_used >= plan.overall_budget:
        reason = "budget_exhausted"
        should_stop = True
    elif queries_attempted >= plan.max_query_count:
        reason = "query_cap_reached"
        should_stop = True
    else:
        reason = "needs_more_evidence"
        should_stop = False
    return StopDecision(
        stop=should_stop,
        reason=reason,
        coverage=coverage,
        intent_coverage=round(intent_coverage, 4),
        source_kind_coverage=round(kind_coverage, 4),
        provider_count=len(providers),
        unique_source_count=len(sources),
        queries_attempted=queries_attempted,
        budget_used=budget_used,
        covered_intents=covered_intents,
        missing_intents=missing_intents,
        missing_source_kinds=missing_kinds,
    )


def evaluate_stop(
    plan: ResearchPlan,
    sources: Sequence[RankedSource],
    *,
    queries_attempted: int,
    budget_used: int,
) -> StopDecision:
    """Evaluate stop criteria independently for incremental executors."""

    return _coverage(plan, sources, int(queries_attempted), int(budget_used))


def rank_sources(
    plan: ResearchPlan,
    results: Mapping[str, Iterable[Source]] | Iterable[SourceHit | Source],
) -> RetrievalResult:
    """Deduplicate and greedily rank bounded evidence with transparent signals."""

    hits, queries_considered = _as_hits(results, plan)
    raw_count = len(hits)
    candidates: dict[str, tuple[SourceHit, float, float, float, float, float, list[str]]] = {}
    for hit in hits:
        key = _canonical_url(hit.source.url)
        relevance, credibility, recency, base = _source_score(hit.query, hit.source, plan.freshness)
        matched = _intent_matches(plan, hit.query, hit.source)
        candidate = (hit, base, relevance, credibility, recency, 0.0, matched)
        previous = candidates.get(key)
        if previous is None:
            candidates[key] = candidate
            continue

        # A URL can be returned by multiple focused intents. Keep the best
        # scoring representation, but retain the union of intent matches so
        # coverage reflects complementary retrieval rather than one duplicate.
        merged_intents = sorted(set(previous[6]) | set(matched))
        if (candidate[1], candidate[2], candidate[3]) > (
            previous[1],
            previous[2],
            previous[3],
        ):
            candidates[key] = (
                candidate[0],
                candidate[1],
                candidate[2],
                candidate[3],
                candidate[4],
                candidate[5],
                merged_intents,
            )
        else:
            candidates[key] = (
                previous[0],
                previous[1],
                previous[2],
                previous[3],
                previous[4],
                previous[5],
                merged_intents,
            )

    deduplicated_count = raw_count - len(candidates)
    remaining = list(candidates.values())
    selected: list[RankedSource] = []
    selected_providers: set[str] = set()
    while remaining:
        scored: list[tuple[float, tuple[SourceHit, float, float, float, float, float, list[str]], float]] = []
        for candidate in remaining:
            hit, base, relevance, credibility, recency, _, matched = candidate
            diversity = 1.0 if hit.source.provider not in selected_providers else 0.0
            selection_score = min(1.0, base + diversity * 0.20)
            scored.append((selection_score, candidate, diversity))
        scored.sort(
            key=lambda item: (
                -item[0],
                -item[1][2],
                -item[1][3],
                item[1][0].source.provider,
                _canonical_url(item[1][0].source.url),
            )
        )
        selection_score, candidate, diversity = scored[0]
        remaining.remove(candidate)
        hit, _, relevance, credibility, recency, _, matched = candidate
        selected_providers.add(hit.source.provider)
        selected.append(
            RankedSource(
                rank=len(selected) + 1,
                source=hit.source,
                score=round(selection_score, 4),
                relevance_score=relevance,
                diversity_score=diversity,
                credibility_score=round(credibility, 4),
                recency_score=recency,
                matched_intents=matched,
            )
        )

    decision = _coverage(plan, selected, queries_considered, raw_count)
    return RetrievalResult(
        plan=plan,
        sources=selected,
        raw_result_count=raw_count,
        deduplicated_count=deduplicated_count,
        queries_considered=queries_considered,
        budget_used=raw_count,
        provider_count=decision.provider_count,
        coverage=decision.coverage,
        covered_intents=decision.covered_intents,
        missing_intents=decision.missing_intents,
        covered_source_kinds=sorted(set(plan.required_source_kinds) - set(decision.missing_source_kinds)),
        missing_source_kinds=decision.missing_source_kinds,
        stop=decision,
    )


class RetrievalPlanner:
    """Small facade for future agent integration and straightforward testing."""

    def create_plan(self, question: str, **kwargs: Any) -> ResearchPlan:
        return build_research_plan(question, **kwargs)

    def rank(
        self,
        plan: ResearchPlan,
        results: Mapping[str, Iterable[Source]] | Iterable[SourceHit | Source],
    ) -> RetrievalResult:
        return rank_sources(plan, results)

    def should_stop(
        self,
        plan: ResearchPlan,
        sources: Sequence[RankedSource],
        *,
        queries_attempted: int,
        budget_used: int,
    ) -> StopDecision:
        return evaluate_stop(
            plan,
            sources,
            queries_attempted=queries_attempted,
            budget_used=budget_used,
        )


__all__ = [
    "FreshnessHint",
    "RankedSource",
    "ResearchPlan",
    "RetrievalPlanner",
    "RetrievalResult",
    "SearchIntent",
    "SourceHit",
    "StopCriteria",
    "StopDecision",
    "build_research_plan",
    "decompose_question",
    "evaluate_stop",
    "rank_sources",
]
