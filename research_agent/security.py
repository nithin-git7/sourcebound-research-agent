"""Small, dependency-free defenses for untrusted retrieved content.

Search results are data, not instructions.  This module keeps the content
available for citation while making the boundary explicit and predictable:
URLs are validated, evidence is bounded, and retrieved text is delimited
before it is returned to a model.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any
from urllib.parse import urlsplit


# These limits are deliberately aligned with the existing Source contract
# where possible.  Evidence is capped after delimiters are included, so the
# resulting record remains valid for the strict Source model.
MAX_URL_LENGTH = 2_000
MAX_EVIDENCE_CHARS = 5_000
MAX_RESPONSE_TEXT_CHARS = 20_000

UNTRUSTED_EVIDENCE_START = "[BEGIN UNTRUSTED EVIDENCE]"
UNTRUSTED_EVIDENCE_END = "[END UNTRUSTED EVIDENCE]"
_DELIMITER_OVERHEAD = len(UNTRUSTED_EVIDENCE_START) + len(UNTRUSTED_EVIDENCE_END) + 2

_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        re.compile(
            r"\b(?:ignore|disregard|forget|override|bypass)\s+"
            r"(?:all\s+|any\s+|the\s+)?"
            r"(?:previous|prior|above|earlier|system|developer|user)\s+"
            r"(?:instructions?|rules?|messages?|prompts?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "instruction_disobedience",
        re.compile(
            r"\b(?:do\s+not|don't)\s+follow\s+"
            r"(?:the\s+)?(?:previous|prior|above|earlier)\s+"
            r"(?:instructions?|rules?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "prompt_exfiltration",
        re.compile(
            r"\b(?:reveal|show|print|repeat|dump|expose)\s+"
            r"(?:the\s+)?(?:system|developer|hidden|secret)\s+"
            r"(?:prompt|instructions?|message)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "role_impersonation",
        re.compile(
            r"\b(?:you\s+are\s+now|act\s+as|pretend\s+to\s+be)\s+"
            r"(?:an?\s+)?(?:system|developer|assistant|administrator|admin)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "instruction_markup",
        re.compile(r"<\s*(?:system|developer|instruction|prompt)\s*>", re.IGNORECASE),
    ),
    (
        "command_execution",
        re.compile(
            r"\b(?:execute|run)\s+(?:the\s+)?(?:following|next)\s+"
            r"(?:instruction|command)\b",
            re.IGNORECASE,
        ),
    ),
)

_EVIDENCE_FIELDS = (
    "title",
    "snippet",
    "evidence_text",
    "evidence",
    "source_evidence",
    "citation_text",
    "summary",
    "description",
    "content",
    "text",
)
_RESPONSE_TEXT_FIELDS = ("output_text", "response_text")


class SecurityBoundaryError(ValueError):
    """Base error for rejected content at the model boundary."""


class InvalidSourceURLError(SecurityBoundaryError):
    """Raised when a source URL is not a bounded HTTP(S) URL."""


@dataclass(frozen=True, slots=True)
class UntrustedText:
    """Bounded retrieved text plus the safe representation for a model."""

    text: str
    safe_text: str
    untrusted: bool
    prompt_injection_detected: bool
    flags: tuple[str, ...]
    truncated: bool

    @property
    def marked_text(self) -> str:
        """Alias useful to callers that think of the delimited form as marked."""

        return self.safe_text


def detect_prompt_injection(text: Any) -> tuple[str, ...]:
    """Return stable flag names for common instruction-injection patterns."""

    candidate = _coerce_text(text)
    # Avoid spending unbounded time scanning an attacker-controlled payload.
    candidate = candidate[:MAX_RESPONSE_TEXT_CHARS]
    return tuple(
        name for name, pattern in _INJECTION_PATTERNS if pattern.search(candidate)
    )


def contains_prompt_injection(text: Any) -> bool:
    """Return whether retrieved text contains a recognized injection pattern."""

    return bool(detect_prompt_injection(text))


def mark_untrusted(
    text: Any,
    *,
    max_chars: int = MAX_EVIDENCE_CHARS,
) -> UntrustedText:
    """Bound and mark retrieved text without deleting its evidentiary content."""

    raw = _unwrap_existing_delimiters(_coerce_text(text))
    flags = detect_prompt_injection(raw)
    body_limit = _body_limit(max_chars)
    bounded = raw[:body_limit]
    safe_body = _escape_delimiters(bounded)
    safe_text = _delimit(safe_body)
    return UntrustedText(
        text=bounded,
        safe_text=safe_text,
        untrusted=True,
        prompt_injection_detected=bool(flags),
        flags=flags,
        truncated=len(raw) > body_limit,
    )


def safely_delimit_evidence(
    text: Any,
    *,
    max_chars: int = MAX_EVIDENCE_CHARS,
) -> str:
    """Return bounded evidence wrapped in non-instruction delimiters."""

    return mark_untrusted(text, max_chars=max_chars).safe_text


# Short aliases make the boundary easy to discover for callers without
# duplicating implementation or weakening the canonical names above.
delimit_untrusted_evidence = safely_delimit_evidence
sanitize_evidence = safely_delimit_evidence


def validate_source_url(url: Any, *, max_length: int = MAX_URL_LENGTH) -> str:
    """Validate and return a safe absolute HTTP(S) source URL.

    ``urlsplit`` is used only for parsing; the explicit checks below reject
    schemes and malformed authority forms that a permissive parser can accept.
    """

    if not isinstance(url, str):
        raise InvalidSourceURLError("source URL must be a string")
    candidate = url.strip()
    if not candidate:
        raise InvalidSourceURLError("source URL is empty")
    if len(candidate) > max_length:
        raise InvalidSourceURLError(
            f"source URL exceeds the {max_length}-character limit"
        )
    if any(character.isspace() or ord(character) < 0x20 or ord(character) == 0x7F for character in candidate):
        raise InvalidSourceURLError("source URL contains whitespace or control characters")
    if "\\" in candidate or any(character in candidate for character in '<>"\''):
        raise InvalidSourceURLError("source URL contains ambiguous or forbidden characters")

    try:
        parsed = urlsplit(candidate)
        scheme = parsed.scheme.casefold()
        hostname = parsed.hostname
        # Accessing port forces validation of malformed numeric ports.
        _ = parsed.port
    except ValueError as exc:
        raise InvalidSourceURLError("source URL is malformed") from exc

    if scheme not in {"http", "https"}:
        raise InvalidSourceURLError("source URL scheme must be http or https")
    if not parsed.netloc or not hostname:
        raise InvalidSourceURLError("source URL must include a host")
    if hostname in {".", ".."}:
        raise InvalidSourceURLError("source URL host is malformed")
    return candidate


def sanitize_source_record(
    record: Mapping[str, Any],
    *,
    max_evidence_chars: int = MAX_EVIDENCE_CHARS,
) -> dict[str, Any]:
    """Sanitize one JSON-like source while retaining IDs and metadata."""

    if not isinstance(record, Mapping):
        raise SecurityBoundaryError("source record must be a mapping")

    sanitized = dict(record)
    sanitized["url"] = validate_source_url(record.get("url"))
    metadata_value = record.get("metadata", {})
    metadata = dict(metadata_value) if isinstance(metadata_value, Mapping) else {}

    flags: set[str] = set()
    bounded_fields: list[str] = []
    truncated = False
    for field in _EVIDENCE_FIELDS:
        if field not in record or record[field] is None:
            continue
        # Titles are also source-controlled text, but the strict Source model
        # caps them at 300 characters. Keep the delimiters inside that field
        # limit so the safety boundary cannot make an otherwise valid record
        # fail validation.
        field_limit = min(max_evidence_chars, 300) if field == "title" else max_evidence_chars
        marked = mark_untrusted(record[field], max_chars=field_limit)
        sanitized[field] = marked.safe_text
        flags.update(marked.flags)
        bounded_fields.append(field)
        truncated = truncated or marked.truncated

    existing_security = metadata.get("security")
    security = dict(existing_security) if isinstance(existing_security, Mapping) else {}
    security.update(
        {
            "untrusted": True,
            "prompt_injection_detected": bool(flags),
            "prompt_injection_flags": sorted(flags),
            "truncated": truncated,
            "bounded_fields": bounded_fields,
            "evidence_delimited": True,
        }
    )
    metadata["security"] = security
    sanitized["metadata"] = metadata
    return sanitized


def sanitize_search_output(
    payload: Any,
    *,
    max_evidence_chars: int = MAX_EVIDENCE_CHARS,
    max_response_chars: int = MAX_RESPONSE_TEXT_CHARS,
) -> Any:
    """Apply the security boundary to router output without changing its shape.

    Invalid source records are rejected individually so one hostile result does
    not erase usable evidence from other providers.  The source collection is
    kept under its original ``sources``/``results`` key, and all other fields
    are preserved.  This keeps normal SearchBundle-shaped output compatible
    with the existing strict report pipeline.
    """

    if isinstance(payload, Mapping):
        output = dict(payload)
        source_keys = [
            key
            for key in ("sources", "results")
            if isinstance(output.get(key), (list, tuple))
        ]
        for key in source_keys:
            sanitized_records: list[dict[str, Any]] = []
            for record in output[key]:
                try:
                    sanitized_records.append(
                        sanitize_source_record(
                            record,
                            max_evidence_chars=max_evidence_chars,
                        )
                    )
                except SecurityBoundaryError:
                    # Fail closed for the individual record while retaining
                    # valid records from the same search fan-out.
                    continue
            output[key] = sanitized_records

        for key in _RESPONSE_TEXT_FIELDS:
            if isinstance(output.get(key), str):
                output[key] = safely_delimit_evidence(
                    output[key],
                    max_chars=max_response_chars,
                )
        return output

    if isinstance(payload, (list, tuple)):
        sanitized_list: list[Any] = []
        for item in payload:
            if not isinstance(item, Mapping):
                continue
            try:
                sanitized_list.append(
                    sanitize_source_record(item, max_evidence_chars=max_evidence_chars)
                )
            except SecurityBoundaryError:
                continue
        return sanitized_list

    return payload


def _coerce_text(value: Any) -> str:
    return "" if value is None else str(value)


def _body_limit(max_chars: int) -> int:
    try:
        limit = int(max_chars)
    except (TypeError, ValueError):
        limit = MAX_EVIDENCE_CHARS
    return max(1, limit - _DELIMITER_OVERHEAD)


def _unwrap_existing_delimiters(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith(UNTRUSTED_EVIDENCE_START) and stripped.endswith(
        UNTRUSTED_EVIDENCE_END
    ):
        return stripped[
            len(UNTRUSTED_EVIDENCE_START) : -len(UNTRUSTED_EVIDENCE_END)
        ].strip()
    return text


def _escape_delimiters(text: str) -> str:
    escaped = re.sub(
        re.escape(UNTRUSTED_EVIDENCE_START),
        "[removed]",
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(
        re.escape(UNTRUSTED_EVIDENCE_END),
        "[removed]",
        escaped,
        flags=re.IGNORECASE,
    )


def _delimit(text: str) -> str:
    return f"{UNTRUSTED_EVIDENCE_START}\n{text}\n{UNTRUSTED_EVIDENCE_END}"


__all__ = [
    "InvalidSourceURLError",
    "MAX_EVIDENCE_CHARS",
    "MAX_RESPONSE_TEXT_CHARS",
    "MAX_URL_LENGTH",
    "SecurityBoundaryError",
    "UNTRUSTED_EVIDENCE_END",
    "UNTRUSTED_EVIDENCE_START",
    "UntrustedText",
    "contains_prompt_injection",
    "delimit_untrusted_evidence",
    "detect_prompt_injection",
    "mark_untrusted",
    "safely_delimit_evidence",
    "sanitize_evidence",
    "sanitize_search_output",
    "sanitize_source_record",
    "validate_source_url",
]
