"""Thread-safe in-memory research job orchestration.

The manager is intentionally storage-agnostic: it provides a small production-
shaped contract for the optional HTTP adapter while keeping the core agent free
from web-framework dependencies.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from enum import Enum
from threading import Lock
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field, StrictStr

from .models import ResearchReport, StrictModel
from .telemetry import RunTelemetry


def _now() -> datetime:
    return datetime.now(timezone.utc)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResearchRequest(StrictModel):
    question: StrictStr = Field(min_length=8, max_length=500)
    mode: Literal["auto", "live", "sample"] = "auto"


class ResearchJob(StrictModel):
    job_id: StrictStr
    status: JobStatus
    question: StrictStr
    mode: Literal["auto", "live", "sample"]
    phase: StrictStr
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    report: ResearchReport | None = None
    error_code: StrictStr | None = None
    error: StrictStr | None = None
    telemetry: dict[str, Any] = Field(default_factory=dict)


ResearchRunner = Callable[[ResearchRequest], ResearchReport]


class ResearchJobManager:
    """Execute bounded research jobs and expose immutable snapshots."""

    def __init__(
        self,
        runner: ResearchRunner,
        *,
        max_workers: int = 2,
        max_jobs: int = 100,
        ttl_seconds: int = 3_600,
    ) -> None:
        if max_workers < 1 or max_jobs < 1 or ttl_seconds < 1:
            raise ValueError("job manager limits must be positive")
        self._runner = runner
        self._max_jobs = max_jobs
        self._ttl = timedelta(seconds=ttl_seconds)
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="sourcebound-job",
        )
        self._lock = Lock()
        self._jobs: dict[str, ResearchJob] = {}
        self._futures: dict[str, Future[None]] = {}

    def submit(self, request: ResearchRequest) -> ResearchJob:
        with self._lock:
            self._prune_locked()
            if len(self._jobs) >= self._max_jobs:
                raise RuntimeError("research job capacity is full")
            job_id = str(uuid4())
            job = ResearchJob(
                job_id=job_id,
                status=JobStatus.QUEUED,
                question=request.question.strip(),
                mode=request.mode,
                phase="waiting for a worker",
                created_at=_now(),
            )
            self._jobs[job_id] = job
            self._futures[job_id] = self._executor.submit(
                self._execute,
                job_id,
                request.model_copy(update={"question": request.question.strip()}),
            )
            return job.model_copy(deep=True)

    def get(self, job_id: str) -> ResearchJob | None:
        with self._lock:
            self._prune_locked()
            job = self._jobs.get(job_id)
            return job.model_copy(deep=True) if job is not None else None

    def cancel(self, job_id: str) -> ResearchJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.status != JobStatus.QUEUED:
                return job.model_copy(deep=True)
            future = self._futures.get(job_id)
            if future is not None and future.cancel():
                self._jobs[job_id] = job.model_copy(
                    update={
                        "status": JobStatus.CANCELLED,
                        "phase": "cancelled before execution",
                        "completed_at": _now(),
                    }
                )
            return self._jobs[job_id].model_copy(deep=True)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _execute(self, job_id: str, request: ResearchRequest) -> None:
        telemetry = RunTelemetry(job_id)
        with self._lock:
            current = self._jobs.get(job_id)
            if current is None or current.status == JobStatus.CANCELLED:
                return
            self._jobs[job_id] = current.model_copy(
                update={
                    "status": JobStatus.RUNNING,
                    "phase": "researching and verifying evidence",
                    "started_at": _now(),
                }
            )
        try:
            with telemetry.timer():
                report = self._runner(request)
        except Exception as exc:
            telemetry.set_stop_reason("failed")
            error_code, message = _public_error(exc)
            with self._lock:
                current = self._jobs[job_id]
                self._jobs[job_id] = current.model_copy(
                    update={
                        "status": JobStatus.FAILED,
                        "phase": "failed",
                        "completed_at": _now(),
                        "error_code": error_code,
                        "error": message,
                        "telemetry": telemetry.snapshot(),
                    }
                )
            return
        for status in report.provider_status:
            provider = telemetry.provider(status.provider_id)
            provider.record_results(status.result_count)
            if not status.ok:
                provider.record_failure()
            provider.set_stop_reason("complete" if status.ok else "failed")
        stop_reason = next(
            (trace.stop_reason for trace in reversed(report.tool_calls) if trace.stop_reason),
            "report_ready",
        )
        telemetry.set_stop_reason(stop_reason)
        with self._lock:
            current = self._jobs[job_id]
            self._jobs[job_id] = current.model_copy(
                update={
                    "status": JobStatus.SUCCEEDED,
                    "phase": "report ready",
                    "completed_at": _now(),
                    "report": report,
                    "telemetry": telemetry.snapshot(),
                }
            )

    def _prune_locked(self) -> None:
        cutoff = _now() - self._ttl
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if job.completed_at is not None and job.completed_at < cutoff
        ]
        for job_id in expired:
            self._jobs.pop(job_id, None)
            self._futures.pop(job_id, None)


def _public_error(error: Exception) -> tuple[str, str]:
    """Map internal failures to stable details safe for an HTTP response."""

    message = str(error).strip()
    if isinstance(error, ValueError) and message:
        return "invalid_request", message[:200]
    if "OPENAI_API_KEY" in message:
        return "configuration_error", (
            "Live research is not configured on this deployment."
        )
    return "research_failed", "The research job failed before a report was produced."


__all__ = [
    "JobStatus",
    "ResearchJob",
    "ResearchJobManager",
    "ResearchRequest",
]
