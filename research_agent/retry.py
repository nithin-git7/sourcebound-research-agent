"""Small dependency-free retry primitive with transient-error handling."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import email.utils
import errno
import math
import random
import re
import time
from numbers import Real
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """Optional named retry settings for callers that prefer a policy object."""

    max_attempts: int = 3
    initial_delay: float = 0.0
    max_delay: float = 60.0
    jitter: float = 0.0


# HTTP responses that are normally safe to retry. Other 4xx responses are
# generally deterministic caller/authentication/input failures.
_RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429})

# Common network/API errno values that can be transient without importing an
# HTTP client or model SDK.
_TRANSIENT_ERRNOS = frozenset(
    {
        errno.EAGAIN,
        errno.ECONNABORTED,
        errno.ECONNREFUSED,
        errno.ECONNRESET,
        errno.EHOSTUNREACH,
        errno.EINTR,
        errno.ENETDOWN,
        errno.ENETRESET,
        errno.ENETUNREACH,
        errno.EPIPE,
        errno.ETIMEDOUT,
    }
)

_RETRYABLE_CLASS_NAMES = frozenset(
    {
        "api_connection_error",
        "api_error",
        "api_timeout_error",
        "bad_gateway_error",
        "busy_error",
        "chunked_encoding_error",
        "connect_error",
        "connect_timeout",
        "connection_error",
        "gateway_timeout_error",
        "internal_server_error",
        "network_error",
        "overloaded_error",
        "pool_timeout",
        "protocol_error",
        "rate_limit_error",
        "read_error",
        "read_timeout",
        "remote_protocol_error",
        "server_error",
        "service_unavailable_error",
        "temporary_error",
        "timeout",
        "timeout_error",
        "timeout_exception",
        "too_many_requests_error",
        "transient_error",
        "throttling_error",
        "write_error",
        "write_timeout",
    }
)

_NON_RETRYABLE_CLASS_NAMES = frozenset(
    {
        "authentication_error",
        "bad_request_error",
        "configuration_error",
        "content_filter_error",
        "forbidden_error",
        "invalid_argument_error",
        "invalid_request_error",
        "malformed_request_error",
        "not_found_error",
        "permission_denied_error",
        "unprocessable_entity_error",
        "unsupported_error",
        "validation_error",
    }
)

_DETERMINISTIC_EXCEPTIONS = (
    TypeError,
    ValueError,
    ImportError,
    ModuleNotFoundError,
    AssertionError,
    AttributeError,
    KeyError,
    IndexError,
    NameError,
    NotImplementedError,
    SyntaxError,
    FileNotFoundError,
    PermissionError,
)

_DETERMINISTIC_MESSAGE_MARKERS = (
    "authentication",
    "bad request",
    "forbidden",
    "invalid argument",
    "invalid input",
    "invalid parameter",
    "malformed request",
    "not found",
    "permission denied",
    "validation error",
    "unauthorized",
    "unsupported",
)

_TRANSIENT_MESSAGE_MARKERS = (
    "bad gateway",
    "connection reset",
    "connection refused",
    "gateway timeout",
    "internal server error",
    "network error",
    "overloaded",
    "rate limit",
    "service unavailable",
    "temporarily unavailable",
    "temporary failure",
    "timed out",
    "timeout",
    "too many requests",
    "try again later",
    "transient",
)


def is_retryable_error(error: BaseException) -> bool:
    """Return whether error represents a transient operation failure.

    The classifier uses standard-library types, status fields, common SDK
    exception names, and conservative message fallbacks. It does not import
    requests, httpx, or a model SDK, so it also works in offline mode.
    """

    if not isinstance(error, BaseException):
        return False
    if isinstance(error, (KeyboardInterrupt, SystemExit, GeneratorExit)):
        return False
    if isinstance(error, _DETERMINISTIC_EXCEPTIONS):
        return False

    class_names = _class_names(error)
    if class_names & _NON_RETRYABLE_CLASS_NAMES:
        return False

    explicit = _explicit_retryable_flag(error)
    if explicit is not None:
        return explicit

    status = _status_code(error)
    if status is not None:
        return status in _RETRYABLE_STATUS_CODES or 500 <= status <= 599

    if get_retry_after_seconds(error) is not None:
        return True

    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    if isinstance(error, OSError) and getattr(error, "errno", None) in _TRANSIENT_ERRNOS:
        return True

    if class_names & _RETRYABLE_CLASS_NAMES:
        return True
    if _class_name_has_transient_marker(class_names):
        return True

    message = str(error).lower()
    if any(marker in message for marker in _DETERMINISTIC_MESSAGE_MARKERS):
        return False
    return any(marker in message for marker in _TRANSIENT_MESSAGE_MARKERS)


def get_retry_after_seconds(error: BaseException) -> float | None:
    """Read a numeric or HTTP-date Retry-After value from an error.

    Both exception attributes (retry_after and retry_after_seconds) and
    response/header shapes are supported. Invalid values are ignored.
    """

    for owner in _related_objects(error):
        for attribute in ("retry_after_seconds", "retry_after"):
            parsed = _parse_retry_after(_safe_get(owner, attribute))
            if parsed is not None:
                return parsed

        headers = _safe_get(owner, "headers")
        parsed = _parse_retry_after(_header_value(headers, "retry-after"))
        if parsed is not None:
            return parsed

    return None


def retry_call(
    function: Callable[..., T],
    *args: Any,
    policy: RetryPolicy | None = None,
    max_attempts: int = 3,
    initial_delay: float = 0.0,
    backoff_factor: float = 2.0,
    max_delay: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
    retry_exceptions: tuple[type[BaseException], ...] = (Exception,),
    max_retries: int | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    **kwargs: Any,
) -> T:
    """Call a function again after retryable failures.

    max_attempts counts the initial call. max_retries and sleep_fn are
    compatibility aliases. max_delay is a direct-call alias for
    RetryPolicy.max_delay and caps generated exponential backoff, including
    jitter. A valid server-provided Retry-After takes precedence over generated
    backoff.
    """

    if max_retries is not None:
        max_attempts = max_retries + 1
    if sleep_fn is not None:
        sleep = sleep_fn
    if policy is not None:
        max_attempts = policy.max_attempts
        initial_delay = policy.initial_delay
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if initial_delay < 0 or backoff_factor < 0:
        raise ValueError("retry delays must not be negative")

    configured_max_delay = policy.max_delay if policy is not None else float("inf")
    if max_delay is not None:
        configured_max_delay = max_delay
    retry_delay_cap = _validated_delay_value(configured_max_delay, "max_delay")
    jitter = policy.jitter if policy is not None else 0.0
    if jitter < 0:
        raise ValueError("retry delay caps and jitter must not be negative")

    delay = float(initial_delay)
    # Preserve the public retry_exceptions extension point for callers that
    # explicitly opt in to a custom transient exception type. The default
    # remains conservative and only retries classified transient failures.
    explicit_exception_allowlist = retry_exceptions != (Exception,)

    for attempt in range(max_attempts):
        try:
            return function(*args, **kwargs)
        except retry_exceptions as exc:
            if not _should_retry(exc, explicit_exception_allowlist):
                raise
            if attempt == max_attempts - 1:
                raise

            retry_after = get_retry_after_seconds(exc)
            if retry_after is not None:
                wait_for = retry_after
            elif delay or jitter:
                wait_for = min(
                    retry_delay_cap,
                    delay + (random.uniform(0, jitter) if jitter else 0.0),
                )
            else:
                wait_for = 0.0

            if wait_for > 0:
                sleep(wait_for)
            delay = min(retry_delay_cap, delay * backoff_factor)

    raise RuntimeError("retry loop did not return or raise")


def _should_retry(error: BaseException, explicit_exception_allowlist: bool) -> bool:
    if is_retryable_error(error):
        return True
    if not explicit_exception_allowlist:
        return False
    # Keep explicit custom exception opt-in behavior, while never retrying
    # deterministic input/programming failures.
    return not isinstance(error, _DETERMINISTIC_EXCEPTIONS)


def _validated_delay_value(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a non-negative number") from None
    if math.isnan(result) or result < 0:
        raise ValueError(f"{name} must be a non-negative number")
    return result


def _class_names(error: BaseException) -> set[str]:
    return {
        _normalize_name(cls.__name__)
        for cls in type(error).__mro__
        if cls.__name__
    }


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _class_name_has_transient_marker(class_names: set[str]) -> bool:
    markers = (
        "connection",
        "network",
        "timeout",
        "rate_limit",
        "ratelimit",
        "too_many_requests",
        "throttl",
        "transient",
        "temporary",
        "unavailable",
        "overloaded",
    )
    return any(marker in name for name in class_names for marker in markers)


def _explicit_retryable_flag(error: BaseException) -> bool | None:
    for attribute in ("retryable", "is_retryable", "should_retry", "transient", "temporary"):
        value = _safe_get(error, attribute)
        if isinstance(value, bool):
            return value
    return None


def _status_code(error: BaseException) -> int | None:
    for owner in _related_objects(error):
        for attribute in ("status_code", "status", "code"):
            parsed = _parse_status(_safe_get(owner, attribute))
            if parsed is not None:
                return parsed
    return None


def _parse_status(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        if isinstance(value, Real):
            number = int(value)
            return number if 100 <= number <= 599 else None
        text = str(value).strip()
        if not re.fullmatch(r"\d{3}", text):
            return None
        number = int(text)
        return number if 100 <= number <= 599 else None
    except (TypeError, ValueError):
        return None


def _related_objects(error: Any):
    yield error
    response = _safe_get(error, "response")
    if response is not None and response is not error:
        yield response


def _safe_get(owner: Any, attribute: str, default: Any = None) -> Any:
    if isinstance(owner, Mapping):
        if attribute in owner:
            return owner[attribute]
        attribute_lower = attribute.lower()
        for key, value in owner.items():
            if str(key).lower() == attribute_lower:
                return value
        return default
    try:
        return getattr(owner, attribute)
    except (AttributeError, TypeError, ValueError):
        return default


def _header_value(headers: Any, name: str) -> Any:
    if headers is None:
        return None
    if isinstance(headers, Mapping):
        for key, value in headers.items():
            if str(key).lower() == name.lower():
                return value
        return None
    getter = _safe_get(headers, "get")
    if callable(getter):
        for candidate in (name, name.title(), name.lower()):
            try:
                value = getter(candidate)
            except (AttributeError, KeyError, TypeError, ValueError):
                value = None
            if value is not None:
                return value
    items = _safe_get(headers, "items")
    if callable(items):
        try:
            for key, value in items():
                if str(key).lower() == name.lower():
                    return value
        except (AttributeError, KeyError, TypeError, ValueError):
            return None
    return None


def _parse_retry_after(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, timedelta):
        seconds = value.total_seconds()
    elif isinstance(value, Real):
        seconds = float(value)
    else:
        try:
            seconds = float(str(value).strip())
        except (TypeError, ValueError):
            seconds = None
        if seconds is None or not math.isfinite(seconds):
            try:
                parsed_date = email.utils.parsedate_to_datetime(str(value).strip())
            except (TypeError, ValueError, OverflowError):
                return None
            if parsed_date is None:
                return None
            if parsed_date.tzinfo is None:
                parsed_date = parsed_date.replace(tzinfo=timezone.utc)
            seconds = parsed_date.timestamp() - datetime.now(timezone.utc).timestamp()
    if not math.isfinite(seconds):
        return None
    return max(0.0, float(seconds))
