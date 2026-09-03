"""Configuration for the research-agent source providers.

The module deliberately keeps configuration dependency-free.  The rest of the
package can therefore be imported in a fixture-only or offline environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping


def _first_env(environ: Mapping[str, str], *names: str) -> str | None:
    for name in names:
        value = environ.get(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _env_bool(environ: Mapping[str, str], names: tuple[str, ...], default: bool) -> bool:
    value = _first_env(environ, *names)
    if value is None:
        return default
    normalized = value.casefold()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _env_int(environ: Mapping[str, str], names: tuple[str, ...], default: int, minimum: int = 1) -> int:
    value = _first_env(environ, *names)
    if value is None:
        return default
    try:
        return max(minimum, int(value))
    except (TypeError, ValueError):
        return default


def _env_float(
    environ: Mapping[str, str], names: tuple[str, ...], default: float, minimum: float = 0.1
) -> float:
    value = _first_env(environ, *names)
    if value is None:
        return default
    try:
        return max(minimum, float(value))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings loaded from environment variables.

    ``RESEARCH_AGENT_*`` names are preferred, with conventional generic names
    accepted for credentials and common HTTP settings.  An OpenAI provider is
    enabled only when a key is present unless explicitly disabled.
    """

    wikipedia_enabled: bool = True
    openalex_enabled: bool = True
    openai_web_enabled: bool = True
    openai_api_key: str | None = None
    openai_model: str = "gpt-5.4-mini"
    model: str = "gpt-5.4-mini"
    max_retries: int = 3
    max_tool_rounds: int = 4
    initial_retry_delay: float = 0.25
    max_retry_delay: float = 4.0
    request_timeout_seconds: float = 10.0
    max_results: int = 5
    max_query_count: int = 4
    retrieval_budget: int = 20
    freshness_mode: str = "any"
    required_source_kinds: tuple[str, ...] = ()
    semantic_verification_enabled: bool = False
    semantic_verification_model: str = "gpt-5.4-mini"
    wikipedia_endpoint: str = "https://en.wikipedia.org/w/api.php"
    openalex_endpoint: str = "https://api.openalex.org/works"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        """Build settings without raising on malformed optional environment values."""

        env = os.environ if environ is None else environ
        api_key = _first_env(env, "RESEARCH_AGENT_OPENAI_API_KEY", "OPENAI_API_KEY")
        openai_enabled = _env_bool(
            env,
            ("RESEARCH_AGENT_OPENAI_WEB_ENABLED", "OPENAI_WEB_SEARCH_ENABLED"),
            default=api_key is not None,
        )
        return cls(
            wikipedia_enabled=_env_bool(env, ("RESEARCH_AGENT_WIKIPEDIA_ENABLED",), True),
            openalex_enabled=_env_bool(env, ("RESEARCH_AGENT_OPENALEX_ENABLED",), True),
            openai_web_enabled=openai_enabled and api_key is not None,
            openai_api_key=api_key,
            openai_model=(
                _first_env(env, "RESEARCH_AGENT_MODEL", "RESEARCH_AGENT_OPENAI_MODEL", "OPENAI_MODEL")
                or "gpt-5.4-mini"
            ),
            model=(
                _first_env(env, "RESEARCH_AGENT_MODEL", "RESEARCH_AGENT_OPENAI_MODEL", "OPENAI_MODEL")
                or "gpt-5.4-mini"
            ),
            max_retries=_env_int(env, ("RESEARCH_MAX_RETRIES", "RESEARCH_AGENT_MAX_RETRIES"), 3, minimum=0),
            max_tool_rounds=_env_int(env, ("RESEARCH_MAX_TOOL_ROUNDS", "RESEARCH_AGENT_MAX_TOOL_ROUNDS"), 4),
            initial_retry_delay=_env_float(
                env, ("RESEARCH_INITIAL_RETRY_DELAY", "RESEARCH_AGENT_INITIAL_RETRY_DELAY"), 0.25, minimum=0.0
            ),
            max_retry_delay=_env_float(
                env, ("RESEARCH_MAX_RETRY_DELAY", "RESEARCH_AGENT_MAX_RETRY_DELAY"), 4.0, minimum=0.0
            ),
            request_timeout_seconds=_env_float(
                env,
                ("RESEARCH_SOURCE_TIMEOUT", "RESEARCH_AGENT_TIMEOUT_SECONDS", "RESEARCH_AGENT_REQUEST_TIMEOUT", "HTTP_TIMEOUT"),
                10.0,
            ),
            max_results=_env_int(env, ("RESEARCH_AGENT_MAX_RESULTS", "MAX_RESULTS"), 5),
            max_query_count=min(
                8,
                _env_int(
                    env,
                    ("RESEARCH_AGENT_MAX_QUERY_COUNT", "RESEARCH_MAX_QUERY_COUNT"),
                    4,
                ),
            ),
            retrieval_budget=min(
                400,
                _env_int(
                    env,
                    ("RESEARCH_AGENT_RETRIEVAL_BUDGET", "RESEARCH_RETRIEVAL_BUDGET"),
                    20,
                ),
            ),
            freshness_mode=(
                _first_env(
                    env,
                    "RESEARCH_AGENT_FRESHNESS_MODE",
                    "RESEARCH_FRESHNESS_MODE",
                )
                or "any"
            ).casefold(),
            required_source_kinds=tuple(
                kind.strip()
                for kind in (
                    _first_env(
                        env,
                        "RESEARCH_AGENT_REQUIRED_SOURCE_KINDS",
                        "RESEARCH_REQUIRED_SOURCE_KINDS",
                    )
                    or ""
                ).split(",")
                if kind.strip()
            ),
            semantic_verification_enabled=_env_bool(
                env,
                ("RESEARCH_AGENT_SEMANTIC_VERIFICATION_ENABLED",),
                False,
            ),
            semantic_verification_model=(
                _first_env(env, "RESEARCH_AGENT_SEMANTIC_VERIFICATION_MODEL")
                or "gpt-5.4-mini"
            ),
            wikipedia_endpoint=(
                _first_env(env, "RESEARCH_AGENT_WIKIPEDIA_ENDPOINT")
                or "https://en.wikipedia.org/w/api.php"
            ),
            openalex_endpoint=(
                _first_env(env, "RESEARCH_AGENT_OPENALEX_ENDPOINT")
                or "https://api.openalex.org/works"
            ),
        )
