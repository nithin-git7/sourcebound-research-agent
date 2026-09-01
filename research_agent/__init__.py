"""Sourcebound: a citation-grounded multi-source research agent."""

from .agent import ResearchAgent
from .config import Settings
from .models import ResearchReport, Source
from .planning import (
    FreshnessHint,
    ResearchPlan,
    RetrievalPlanner,
    RetrievalResult,
    build_research_plan,
)
from .retry import RetryPolicy, get_retry_after_seconds, is_retryable_error
from .security import (
    SecurityBoundaryError,
    contains_prompt_injection,
    mark_untrusted,
    sanitize_search_output,
    validate_source_url,
)
from .verification import (
    DeterministicLexicalVerifier,
    EvidenceVerdict,
    EvidenceVerificationReport,
    verify_evidence,
)

__all__ = [
    "DeterministicLexicalVerifier",
    "EvidenceVerdict",
    "EvidenceVerificationReport",
    "FreshnessHint",
    "ResearchAgent",
    "ResearchPlan",
    "ResearchReport",
    "RetrievalPlanner",
    "RetrievalResult",
    "RetryPolicy",
    "SecurityBoundaryError",
    "Settings",
    "Source",
    "build_research_plan",
    "contains_prompt_injection",
    "get_retry_after_seconds",
    "is_retryable_error",
    "mark_untrusted",
    "sanitize_search_output",
    "validate_source_url",
    "verify_evidence",
]
