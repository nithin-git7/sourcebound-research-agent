from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import ResearchAgent
from .config import Settings
from .evaluation import run_evaluation_suite
from .llm import OpenAIResponsesAdapter
from .offline import build_offline_agent
from .rendering import render_markdown
from .retry import RetryPolicy
from .sources import MultiSourceSearchTool, providers_from_settings


def _build_live_agent(settings: Settings) -> ResearchAgent:
    model = OpenAIResponsesAdapter(api_key=settings.openai_api_key)
    semantic_verifier = None
    if settings.semantic_verification_enabled:
        from .semantic_verification import SemanticEvidenceVerifier

        semantic_verifier = SemanticEvidenceVerifier(
            model,
            model=settings.semantic_verification_model,
        )
    providers = providers_from_settings(settings)
    if not providers:
        raise RuntimeError("No live source providers are enabled.")
    search_tool = MultiSourceSearchTool(
        providers,
        retry_policy=RetryPolicy(
            max_attempts=max(1, settings.max_retries + 1),
            initial_delay=settings.initial_retry_delay,
            max_delay=settings.max_retry_delay,
        ),
    )
    return ResearchAgent(
        llm=model,
        search_tool=search_tool,
        model_name=settings.model,
        max_turns=max(1, settings.max_tool_rounds),
        max_attempts=max(1, settings.max_retries + 1),
        retry_initial_delay=settings.initial_retry_delay,
        settings=settings,
        semantic_verifier=semantic_verifier,
    )


def _write_or_print(content: str, output: str | None) -> None:
    if output:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        print(f"Wrote {destination}")
    else:
        print(content)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research-agent",
        description="Search, compare, and cite independent sources.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    research = subparsers.add_parser("research", aliases=["ask"], help="Run a research report.")
    research.add_argument("question", help="Question to research.")
    research.add_argument("--offline", action="store_true", help="Use deterministic fixtures.")
    research.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="markdown",
        help="Output format.",
    )
    research.add_argument("--output", help="Optional file path for the report.")

    evaluate = subparsers.add_parser("evaluate", help="Run the deterministic evaluation suite.")
    evaluate.add_argument("--output", help="Optional file path for evaluation JSON.")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in {"research", "ask"}:
        settings = Settings.from_env()
        use_offline = bool(args.offline or not settings.openai_api_key)
        if use_offline:
            if not args.offline:
                print(
                    "OPENAI_API_KEY is not set; using deterministic offline fixtures.",
                    file=sys.stderr,
                )
            agent = build_offline_agent()
        else:
            try:
                agent = _build_live_agent(settings)
            except RuntimeError as error:
                print(str(error), file=sys.stderr)
                return 2

        try:
            report = agent.run(args.question)
        except Exception as error:
            print(f"Research failed: {error}", file=sys.stderr)
            return 1

        if args.format == "json":
            content = json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False)
        else:
            content = render_markdown(report)
        _write_or_print(content, args.output)
        return 0

    if args.command == "evaluate":
        try:
            suite = run_evaluation_suite()
        except Exception as error:
            print(f"Evaluation failed: {error}", file=sys.stderr)
            return 1
        content = json.dumps(suite.model_dump(mode="json"), indent=2, ensure_ascii=False)
        _write_or_print(content, args.output)
        return 0 if suite.passed else 1

    return 2
