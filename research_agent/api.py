"""Optional HTTP adapter for the Sourcebound research report contract.

The core package intentionally has no web-framework dependency. Importing this
module is safe in the minimal installation; FastAPI is imported only when
create_app is called. The default report is the deterministic portfolio
fixture when the source tree is available, with an offline-generated fallback
for installed packages.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

from .models import ResearchReport
from .jobs import JobStatus, ResearchJobManager, ResearchRequest


REPORT_PATH_ENV = "SOURCEBOUND_REPORT_PATH"
DEFAULT_QUESTION = "What are the trade-offs of retrieval-augmented generation?"
DEFAULT_RUN_ID = "portfolio-demo-001"
DEFAULT_GENERATED_AT = "2026-01-15T09:00:00+00:00"


def _candidate_report_paths(report_path: str | Path | None) -> list[Path]:
    candidates: list[Path] = []
    if report_path is not None:
        candidates.append(Path(report_path))
    configured_path = os.environ.get(REPORT_PATH_ENV)
    if configured_path:
        candidates.append(Path(configured_path))
    candidates.extend(
        [
            Path.cwd() / "portfolio" / "sample_report.json",
            Path(__file__).resolve().parents[1] / "portfolio" / "sample_report.json",
        ]
    )

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.expanduser().resolve(strict=False)).casefold()
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _coerce_report(value: ResearchReport | Mapping[str, Any]) -> ResearchReport:
    if isinstance(value, ResearchReport):
        return value
    payload = dict(value)
    verification = payload.get("verification")
    if isinstance(verification, Mapping):
        verification_payload = dict(verification)
        checks = verification_payload.get("claim_checks")
        if isinstance(checks, list):
            from .verification import EvidenceVerdict

            verification_payload["claim_checks"] = [
                (
                    {
                        **check,
                        "verdict": EvidenceVerdict(check["verdict"]),
                    }
                    if isinstance(check, Mapping)
                    and isinstance(check.get("verdict"), str)
                    else check
                )
                for check in checks
            ]
        payload["verification"] = verification_payload
    return ResearchReport.model_validate(payload)


def _read_report(path: Path) -> ResearchReport:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _coerce_report(payload)


def _fallback_report() -> ResearchReport:
    """Build a stable offline response when the portfolio fixture is absent."""

    from .offline import build_offline_agent

    generated = build_offline_agent().run(DEFAULT_QUESTION)
    payload = generated.model_dump(mode="json")
    payload["run_id"] = DEFAULT_RUN_ID
    payload["generated_at"] = DEFAULT_GENERATED_AT
    # The strict Python boundary expects enum instances, while the JSON
    # boundary intentionally accepts their wire values. Round-trip through the
    # JSON contract for the installed-package fallback.
    return ResearchReport.model_validate_json(json.dumps(payload))


def _load_report(
    *,
    report_path: str | Path | None,
    report: ResearchReport | Mapping[str, Any] | None,
) -> ResearchReport:
    if report is not None:
        return _coerce_report(report)
    for candidate in _candidate_report_paths(report_path):
        if not candidate.is_file():
            continue
        try:
            return _read_report(candidate)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return _fallback_report()


def create_app(
    *,
    report_path: str | Path | None = None,
    report: ResearchReport | Mapping[str, Any] | None = None,
    research_runner: Callable[[ResearchRequest], ResearchReport] | None = None,
    job_manager: ResearchJobManager | None = None,
) -> Any:
    """Create the optional FastAPI application.

    fastapi and uvicorn are intentionally not core dependencies. Install them
    separately for the web adapter, for example:
    python -m pip install fastapi uvicorn

    The app serves a validated report at /report and the same payload at
    /api/report for front-end clients. report_path or report can be supplied by
    a deployment; otherwise the portfolio sample is discovered.
    """

    try:
        from fastapi import FastAPI
    except ModuleNotFoundError as exc:
        if exc.name == "fastapi" or (exc.name and exc.name.startswith("fastapi.")):
            raise RuntimeError(
                "The optional API requires FastAPI. Install it with "
                "'python -m pip install fastapi uvicorn'."
            ) from exc
        raise

    cached_report: ResearchReport | None = None

    def default_runner(request: ResearchRequest) -> ResearchReport:
        if request.mode == "sample":
            if request.question.strip().casefold() != DEFAULT_QUESTION.casefold():
                raise ValueError(
                    "Sample mode only supports the bundled RAG demonstration question."
                )
            return _fallback_report()
        from .config import Settings

        settings = Settings.from_env()
        if not settings.openai_api_key:
            raise RuntimeError(
                "Live research requires RESEARCH_AGENT_OPENAI_API_KEY or OPENAI_API_KEY."
            )
        from .cli import _build_live_agent

        return _build_live_agent(settings).run(request.question)

    owns_manager = job_manager is None
    manager = job_manager or ResearchJobManager(research_runner or default_runner)

    @asynccontextmanager
    async def lifespan(_app: Any):
        try:
            yield
        finally:
            if owns_manager:
                manager.close()

    def get_report() -> dict[str, Any]:
        nonlocal cached_report
        if cached_report is None:
            cached_report = _load_report(report_path=report_path, report=report)
        return cached_report.model_dump(mode="json")

    app = FastAPI(
        title="Sourcebound Research Agent",
        version="0.2.0",
        description="A citation-grounded research trace for the Evidence Lab portfolio.",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "service": "sourcebound-research-agent",
            "version": "0.2.0",
        }

    @app.get("/report")
    def report_endpoint() -> dict[str, Any]:
        return get_report()

    @app.get("/api/report", include_in_schema=False)
    def api_report_endpoint() -> dict[str, Any]:
        return get_report()

    @app.post("/research", status_code=202)
    def create_research_job(request: ResearchRequest) -> dict[str, Any]:
        from fastapi import HTTPException

        try:
            job = manager.submit(request)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return job.model_dump(mode="json")

    @app.get("/research/{job_id}")
    def get_research_job(job_id: str) -> dict[str, Any]:
        from fastapi import HTTPException

        job = manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="research job not found")
        return job.model_dump(mode="json")

    @app.post("/research/{job_id}/cancel")
    def cancel_research_job(job_id: str) -> dict[str, Any]:
        from fastapi import HTTPException

        job = manager.cancel(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="research job not found")
        if job.status not in {JobStatus.QUEUED, JobStatus.CANCELLED}:
            raise HTTPException(
                status_code=409,
                detail=f"job cannot be cancelled while {job.status.value}",
            )
        return job.model_dump(mode="json")

    return app


__all__ = ["create_app"]
