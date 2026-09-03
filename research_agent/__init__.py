"""Sourcebound: a citation-grounded multi-source research agent."""

from .agent import ResearchAgent
from .config import Settings
from .jobs import JobStatus, ResearchJob, ResearchJobManager, ResearchRequest
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
from .semantic_verification import (
    SemanticEvidenceVerifier,
    SemanticVerdict,
    SemanticVerificationReport,
)
from .telemetry import RunTelemetry
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
    "JobStatus",
    "ResearchAgent",
    "ResearchJob",
    "ResearchJobManager",
    "ResearchPlan",
    "ResearchReport",
    "ResearchRequest",
    "RetrievalPlanner",
    "RetrievalResult",
    "RetryPolicy",
    "SecurityBoundaryError",
    "SemanticEvidenceVerifier",
    "SemanticVerdict",
    "SemanticVerificationReport",
    "Settings",
    "Source",
    "RunTelemetry",
    "build_research_plan",
    "contains_prompt_injection",
    "get_retry_after_seconds",
    "is_retryable_error",
    "mark_untrusted",
    "sanitize_search_output",
    "validate_source_url",
    "verify_evidence",
]

__version__ = "0.2.0"
