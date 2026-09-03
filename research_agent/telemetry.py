"""Thread-safe, dependency-free telemetry primitives for research runs."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
import math
from numbers import Real
import threading
import time
from types import TracebackType
from typing import Any, Self


Clock = Callable[[], float]


@dataclass
class _Metrics:
    latency_seconds: float = 0.0
    retries: int = 0
    failures: int = 0
    result_count: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    stop_reason: str | None = None

    def snapshot(self) -> dict[str, Any]:
        """Return a detached dictionary containing JSON-compatible values."""

        return {
            "latency_ms": round(self.latency_seconds * 1_000.0, 3),
            "retries": self.retries,
            "failures": self.failures,
            "result_count": self.result_count,
            "cache_hits": self.cache_hits,
            "token_usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.input_tokens + self.output_tokens,
            },
            "estimated_cost_usd": round(self.estimated_cost_usd, 8),
            "stop_reason": self.stop_reason,
        }


class RunTelemetry:
    """Collect telemetry for one research run and each of its providers.

    Counter and usage events recorded for a provider also roll up into the run
    totals. Latency is intentionally different: a run timer measures end-to-end
    wall time, while provider timers measure provider spans only. This avoids
    inflating run latency when provider requests execute concurrently.
    """

    def __init__(self, run_id: str, *, clock: Clock = time.perf_counter) -> None:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.run_id = run_id
        self._clock = clock
        self._lock = threading.RLock()
        self._run = _Metrics()
        self._providers: dict[str, _Metrics] = {}

    def provider(self, provider_id: str) -> ProviderTelemetry:
        """Return a lightweight recorder bound to ``provider_id``."""

        provider_id = _provider_id(provider_id)
        with self._lock:
            self._providers.setdefault(provider_id, _Metrics())
        return ProviderTelemetry(self, provider_id)

    def record_latency(
        self,
        seconds: float,
        *,
        provider_id: str | None = None,
    ) -> None:
        """Record wall time for the run or an individual provider span."""

        value = _nonnegative_number(seconds, "seconds")
        with self._lock:
            self._metrics(provider_id).latency_seconds += value

    def record_retry(self, *, provider_id: str | None = None, count: int = 1) -> None:
        self._increment("retries", count, provider_id)

    def record_failure(self, *, provider_id: str | None = None, count: int = 1) -> None:
        self._increment("failures", count, provider_id)

    def record_results(self, count: int, *, provider_id: str | None = None) -> None:
        self._increment("result_count", count, provider_id)

    def record_cache_hit(self, *, provider_id: str | None = None, count: int = 1) -> None:
        self._increment("cache_hits", count, provider_id)

    def record_tokens(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        provider_id: str | None = None,
    ) -> None:
        input_value = _nonnegative_int(input_tokens, "input_tokens")
        output_value = _nonnegative_int(output_tokens, "output_tokens")
        with self._lock:
            target = self._metrics(provider_id)
            target.input_tokens += input_value
            target.output_tokens += output_value
            if provider_id is not None:
                self._run.input_tokens += input_value
                self._run.output_tokens += output_value

    def record_cost(self, usd: float, *, provider_id: str | None = None) -> None:
        value = _nonnegative_number(usd, "usd")
        with self._lock:
            self._metrics(provider_id).estimated_cost_usd += value
            if provider_id is not None:
                self._run.estimated_cost_usd += value

    def set_stop_reason(
        self,
        reason: str | None,
        *,
        provider_id: str | None = None,
    ) -> None:
        if reason is not None and (not isinstance(reason, str) or not reason.strip()):
            raise ValueError("reason must be None or a non-empty string")
        with self._lock:
            self._metrics(provider_id).stop_reason = reason

    def timer(
        self,
        *,
        provider_id: str | None = None,
        record_failure: bool = True,
    ) -> TelemetryTimer:
        """Create a one-shot timer usable as a context manager.

        By default, an exception leaving the context increments the relevant
        failure count and is re-raised. Call ``stop()`` directly when a context
        manager is inconvenient.
        """

        if provider_id is not None:
            provider_id = _provider_id(provider_id)
        if not isinstance(record_failure, bool):
            raise TypeError("record_failure must be a bool")
        return TelemetryTimer(
            self,
            provider_id=provider_id,
            record_failure=record_failure,
            clock=self._clock,
        )

    def snapshot(self) -> dict[str, Any]:
        """Return an atomic, detached and JSON-serializable snapshot."""

        with self._lock:
            result = self._run.snapshot()
            result["run_id"] = self.run_id
            result["providers"] = {
                provider_id: metrics.snapshot()
                for provider_id, metrics in sorted(self._providers.items())
            }
            return result

    def _metrics(self, provider_id: str | None) -> _Metrics:
        if provider_id is None:
            return self._run
        normalized = _provider_id(provider_id)
        return self._providers.setdefault(normalized, _Metrics())

    def _increment(self, field: str, count: int, provider_id: str | None) -> None:
        value = _nonnegative_int(count, "count")
        with self._lock:
            target = self._metrics(provider_id)
            setattr(target, field, getattr(target, field) + value)
            if provider_id is not None:
                setattr(self._run, field, getattr(self._run, field) + value)


class ProviderTelemetry:
    """Provider-bound facade over a :class:`RunTelemetry` collector."""

    def __init__(self, run: RunTelemetry, provider_id: str) -> None:
        self._run = run
        self.provider_id = provider_id

    def record_latency(self, seconds: float) -> None:
        self._run.record_latency(seconds, provider_id=self.provider_id)

    def record_retry(self, count: int = 1) -> None:
        self._run.record_retry(provider_id=self.provider_id, count=count)

    def record_failure(self, count: int = 1) -> None:
        self._run.record_failure(provider_id=self.provider_id, count=count)

    def record_results(self, count: int) -> None:
        self._run.record_results(count, provider_id=self.provider_id)

    def record_cache_hit(self, count: int = 1) -> None:
        self._run.record_cache_hit(provider_id=self.provider_id, count=count)

    def record_tokens(self, *, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self._run.record_tokens(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            provider_id=self.provider_id,
        )

    def record_cost(self, usd: float) -> None:
        self._run.record_cost(usd, provider_id=self.provider_id)

    def set_stop_reason(self, reason: str | None) -> None:
        self._run.set_stop_reason(reason, provider_id=self.provider_id)

    def timer(self, *, record_failure: bool = True) -> TelemetryTimer:
        return self._run.timer(
            provider_id=self.provider_id,
            record_failure=record_failure,
        )

    def snapshot(self) -> dict[str, Any]:
        """Return this provider's detached snapshot."""

        with self._run._lock:
            return self._run._metrics(self.provider_id).snapshot()


class TelemetryTimer(AbstractContextManager["TelemetryTimer"]):
    """A one-shot monotonic timer that records elapsed telemetry."""

    def __init__(
        self,
        telemetry: RunTelemetry,
        *,
        provider_id: str | None,
        record_failure: bool,
        clock: Clock,
    ) -> None:
        self._telemetry = telemetry
        self._provider_id = provider_id
        self._record_failure = record_failure
        self._clock = clock
        self._started_at: float | None = None
        self._elapsed_seconds: float | None = None
        self._state_lock = threading.Lock()

    @property
    def elapsed_seconds(self) -> float | None:
        return self._elapsed_seconds

    def start(self) -> Self:
        with self._state_lock:
            if self._started_at is not None or self._elapsed_seconds is not None:
                raise RuntimeError("telemetry timers are one-shot")
            self._started_at = self._clock()
        return self

    def stop(self) -> float:
        with self._state_lock:
            if self._started_at is None:
                raise RuntimeError("timer has not been started")
            elapsed = max(0.0, self._clock() - self._started_at)
            self._started_at = None
            self._elapsed_seconds = elapsed
        self._telemetry.record_latency(elapsed, provider_id=self._provider_id)
        return elapsed

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        self.stop()
        if exc_type is not None and self._record_failure:
            self._telemetry.record_failure(provider_id=self._provider_id)
        return False


def _provider_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("provider_id must be a non-empty string")
    return value


def _nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _nonnegative_number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite non-negative number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return number


__all__ = ["ProviderTelemetry", "RunTelemetry", "TelemetryTimer"]
