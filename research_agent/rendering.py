from __future__ import annotations

from .models import ResearchReport
from .security import UNTRUSTED_EVIDENCE_END, UNTRUSTED_EVIDENCE_START


def _display_source_text(value: str) -> str:
    """Remove internal safety wrappers when rendering human-facing Markdown."""

    text = str(value or "").strip()
    if text.startswith(UNTRUSTED_EVIDENCE_START) and text.endswith(
        UNTRUSTED_EVIDENCE_END
    ):
        text = text[
            len(UNTRUSTED_EVIDENCE_START) : -len(UNTRUSTED_EVIDENCE_END)
        ].strip()
    return text


def _markdown_link_text(value: str) -> str:
    return (
        _display_source_text(value)
        .replace("\\", "\\\\")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )


def render_markdown(report: ResearchReport) -> str:
    lines = [
        f"# Research report: {report.question}",
        "",
        f"> Run {report.run_id} | model {report.model} | generated {report.generated_at}",
        "",
        "## Executive summary",
        "",
        report.executive_summary,
        "",
        "## Key findings",
        "",
    ]
    for finding in report.key_findings:
        citations = " ".join(f"[{source_id}]" for source_id in finding.citation_ids)
        lines.append(
            f"- **{finding.finding_id} ({finding.importance})** - {finding.statement} "
            f"(confidence {finding.confidence:.0%}) {citations}"
        )

    lines.extend(["", "## Cross-source comparison", ""])
    for point in report.comparison:
        lines.extend([f"### {point.dimension}", "", point.consensus, ""])
        if point.disagreements:
            lines.append("Differences:")
            lines.extend(f"- {difference}" for difference in point.disagreements)
            lines.append("")
        lines.extend(
            f"- [{view.source_id}] {view.position}" for view in point.source_views
        )
        lines.append("")

    lines.extend(["## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in report.limitations)

    planned_traces = [trace for trace in report.tool_calls if trace.planned_queries]
    if planned_traces:
        lines.extend(["", "## Retrieval plan", ""])
        for trace in planned_traces:
            lines.append(f"- Planned queries: {len(trace.planned_queries)}")
            lines.extend(f"  - {query}" for query in trace.planned_queries)
            if trace.retrieval_coverage is not None:
                lines.append(f"- Coverage: {trace.retrieval_coverage:.0%}")
            if trace.stop_reason:
                lines.append(f"- Stop reason: `{trace.stop_reason}`")
            if trace.missing_intents:
                lines.append(
                    "- Missing retrieval intents: "
                    + ", ".join(trace.missing_intents)
                )

    lines.extend(
        [
            "",
            "## Evidence audit",
            "",
            f"- Score: **{report.audit.score:.0%}**",
            f"- Citation coverage: {report.audit.citation_coverage:.0%}",
            f"- Grounding: {report.audit.grounding_score:.0%}",
            f"- Source diversity: {report.audit.source_diversity:.0%} "
            f"({report.audit.provider_count} providers)",
            f"- Comparison quality: {report.audit.comparison_quality:.0%}",
        ]
    )
    if report.audit.warnings:
        lines.append("- Warnings: " + " ".join(report.audit.warnings))

    lines.extend(["", "## Sources", ""])
    for source in report.sources:
        author_text = f" - {', '.join(source.authors)}" if source.authors else ""
        published_text = f" ({source.published_at})" if source.published_at else ""
        title = _markdown_link_text(source.title)
        snippet = _display_source_text(source.snippet).replace("\n", " ")
        lines.append(
            f"- **[{source.id}]** [{title}](<{source.url}>) - "
            f"{source.provider}/{source.kind}{published_text}{author_text}"
        )
        lines.append(f"  - {snippet[:500]}")
        if source.start_index is not None and source.end_index is not None:
            lines.append(
                f"  - Hosted-response evidence span: "
                f"{source.start_index}:{source.end_index}"
            )

    verification = (
        report.verification.model_dump(mode="json")
        if report.verification is not None
        else {}
    )
    if verification:
        lines.extend(
            [
                "",
                "## Claim evidence verification",
                "",
                f"- Method: {verification.get('method', 'unknown')}",
                f"- Claim coverage: {float(verification.get('claim_coverage', 0)):.0%}",
                f"- Citation coverage: {float(verification.get('citation_coverage', 0)):.0%}",
                "- Supported / partial / unsupported / contradicted: "
                f"{verification.get('supported_claim_count', 0)} / "
                f"{verification.get('partial_claim_count', 0)} / "
                f"{verification.get('unsupported_claim_count', 0)} / "
                f"{verification.get('contradicted_claim_count', 0)}",
            ]
        )
        for check in verification.get("claim_checks", []):
            lines.append(
                f"- `{check.get('claim_id', 'unknown')}`: "
                f"**{check.get('verdict', 'unknown')}** "
                f"({float(check.get('coverage', 0)):.0%})"
            )

    lines.extend(
        [
            "",
            "## Provider status",
            "",
        ]
    )
    for status in report.provider_status:
        if status.ok:
            lines.append(f"- {status.provider}: ok ({status.result_count} sources)")
        else:
            lines.append(f"- {status.provider}: error ({status.error or 'unknown error'})")

    lines.extend(
        [
            "",
            "Citation IDs are application-owned source records; the model cannot "
            "introduce a citation that was not returned by a provider.",
        ]
    )
    return "\n".join(lines).strip() + "\n"
