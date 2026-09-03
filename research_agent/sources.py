"""Search-source providers and the concurrent source fan-out tool."""

from __future__ import annotations

import hashlib
import html
import inspect
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, fields, is_dataclass
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable
from urllib.parse import quote, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

try:  # The planned package provides Source; keep the other imports independently optional.
    from .models import Source
except ImportError:  # pragma: no cover - exercised only before models.py is added.
    @dataclass(frozen=True, slots=True)
    class Source:
        id: str
        title: str
        url: str
        snippet: str = ""

try:
    from .models import ProviderStatus
except ImportError:  # pragma: no cover - exercised only before models.py is added.
    @dataclass(frozen=True, slots=True)
    class ProviderStatus:
        provider_id: str
        provider: str
        ok: bool
        result_count: int = 0
        error: str | None = None

try:
    from .models import SearchBundle
except ImportError:  # pragma: no cover - exercised only before models.py is added.
    @dataclass(frozen=True, slots=True)
    class SearchBundle:
        query: str
        sources: list[Source]
        provider_statuses: list[ProviderStatus]

        @property
        def provider_status(self) -> list[ProviderStatus]:
            return self.provider_statuses

try:
    from .retry import retry_call
except ImportError:  # pragma: no cover - only used while the planned retry.py is absent.
    def retry_call(function: Any, retry_policy: Any = None) -> Any:
        return function()


WIKIPEDIA_SOURCE_ID = "wikipedia"
OPENALEX_SOURCE_ID = "openalex"
OPENAI_WEB_SOURCE_ID = "openai_web_search"
FIXTURE_SOURCE_ID = "fixture"

# Short aliases are useful to callers that prefer provider terminology.
WIKIPEDIA_ID = WIKIPEDIA_SOURCE_ID
OPENALEX_ID = OPENALEX_SOURCE_ID
OPENAI_ID = OPENAI_WEB_SOURCE_ID
FIXTURE_ID = FIXTURE_SOURCE_ID


@runtime_checkable
class SourceProvider(Protocol):
    """Small structural interface implemented by every search provider."""

    source_id: str

    def search(self, query: str, *, limit: int = 5) -> Sequence[Source]:
        ...


def _provider_id(provider: Any) -> str:
    value = getattr(provider, "source_id", None) or getattr(provider, "provider_id", None)
    if value is None:
        value = getattr(provider, "id", None)
    return str(value or provider.__class__.__name__.casefold())


def _field_names(model_type: Any) -> set[str]:
    model_fields = getattr(model_type, "model_fields", None)
    if isinstance(model_fields, Mapping):
        return set(model_fields)
    legacy_fields = getattr(model_type, "__fields__", None)
    if isinstance(legacy_fields, Mapping):
        return set(legacy_fields)
    if is_dataclass(model_type):
        return {field.name for field in fields(model_type)}
    try:
        return {
            name
            for name, parameter in inspect.signature(model_type).parameters.items()
            if name != "self" and parameter.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
    except (TypeError, ValueError):
        return set()


def _make_model(model_type: Any, payload: Mapping[str, Any]) -> Any:
    """Pass only supported fields, while tolerating simple model variations."""

    names = _field_names(model_type)
    selected = {name: value for name, value in payload.items() if name in names}
    try:
        return model_type(**(selected if names else dict(payload)))
    except TypeError:
        # A Pydantic model with aliases or a permissive dataclass may reject the
        # introspected subset; the full payload gives it a second chance.
        return model_type(**dict(payload))


def _make_source(
    provider_id: str,
    title: Any,
    url: Any,
    snippet: Any = "",
    *,
    kind: str = "web",
    published_at: Any = None,
    authors: Sequence[Any] | None = None,
    credibility: str = "unknown",
    evidence: Any = None,
    start_index: Any = None,
    end_index: Any = None,
    metadata: Mapping[str, Any] | None = None,
) -> Source:
    clean_url = str(url or "").strip()
    clean_title = str(title or clean_url or "Untitled").strip()
    clean_evidence = _clean_text(evidence)
    clean_snippet = clean_evidence or _clean_text(snippet) or clean_title or clean_url
    normalized_credibility = str(credibility or "unknown").strip().casefold()
    if normalized_credibility not in {"high", "medium", "unknown"}:
        normalized_credibility = "unknown"
    normalized_authors = [str(author).strip() for author in (authors or []) if str(author).strip()]
    optional_metadata = dict(metadata or {})
    payload = {
        "id": stable_source_id(provider_id, clean_url),
        "source_id": provider_id,
        "provider": provider_id,
        "provider_id": provider_id,
        "title": clean_title,
        "url": clean_url,
        "snippet": clean_snippet,
        "summary": clean_snippet,
        "description": clean_snippet,
        "content": clean_snippet,
        "kind": kind,
        "published_at": str(published_at or "").strip() or None,
        "authors": normalized_authors,
        "credibility": normalized_credibility,
        "publisher": provider_id,
        # These are filtered by _make_model so lean compatible source models
        # still work, while the current strict Source contract retains the
        # canonical evidence and provenance fields.
        "evidence": clean_evidence or clean_snippet,
        "evidence_text": clean_evidence or clean_snippet,
        "citation_text": clean_evidence or clean_snippet,
        "source_evidence": clean_evidence or clean_snippet,
        "start_index": start_index if isinstance(start_index, int) else None,
        "end_index": end_index if isinstance(end_index, int) else None,
        "citation_start": start_index if isinstance(start_index, int) else None,
        "citation_end": end_index if isinstance(end_index, int) else None,
        "metadata": optional_metadata,
        "provenance": optional_metadata,
        "source_metadata": optional_metadata,
    }
    return _make_model(Source, payload)


def _make_status(provider_id: str, ok: bool, count: int = 0, error: str | None = None) -> ProviderStatus:
    payload = {
        "provider_id": provider_id,
        "source_id": provider_id,
        "provider": provider_id,
        "ok": ok,
        "success": ok,
        "result_count": count,
        "count": count,
        "error": error,
        "error_message": error,
    }
    return _make_model(ProviderStatus, payload)


def _make_bundle(query: str, sources: list[Source], statuses: list[ProviderStatus]) -> SearchBundle:
    payload = {
        "query": query,
        "sources": sources,
        "results": sources,
        "provider_statuses": statuses,
        "provider_status": statuses,
        "statuses": statuses,
    }
    return _make_model(SearchBundle, payload)


def _clean_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()


def _limit(value: int) -> int:
    try:
        return max(1, min(int(value), 100))
    except (TypeError, ValueError):
        return 5


def _canonical_url(url: str) -> str:
    """Canonicalize enough URL detail for stable IDs and duplicate suppression."""

    raw = str(url or "").strip()
    parsed = urlsplit(raw)
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
        return raw
    host = (parsed.hostname or "").casefold()
    try:
        port = parsed.port
    except ValueError:
        port = None
    if port and not ((parsed.scheme.casefold() == "http" and port == 80) or (parsed.scheme.casefold() == "https" and port == 443)):
        host = f"{host}:{port}"
    path = parsed.path or "/"
    if len(path) > 1:
        path = path.rstrip("/")
    return urlunsplit((parsed.scheme.casefold(), host, path, parsed.query, ""))


def stable_source_id(provider_id: str, url: str) -> str:
    """Return a deterministic record ID for a provider and canonical URL."""

    digest = hashlib.sha256(_canonical_url(url).encode("utf-8")).hexdigest()[:16]
    return f"{provider_id}:{digest}"


def _http_get_json(url: str, timeout_seconds: float, headers: Mapping[str, str] | None = None) -> Any:
    request = Request(url, headers={"Accept": "application/json", **dict(headers or {})})
    with urlopen(request, timeout=timeout_seconds) as response:  # nosec B310 - URL is provider-controlled.
        return json.loads(response.read().decode("utf-8"))


@dataclass(slots=True)
class WikipediaProvider:
    timeout_seconds: float = 10.0
    endpoint: str = "https://en.wikipedia.org/w/api.php"
    source_id: str = WIKIPEDIA_SOURCE_ID

    def search(self, query: str, *, limit: int = 5) -> list[Source]:
        params = urlencode(
            {"action": "query", "list": "search", "srsearch": str(query), "srlimit": _limit(limit), "format": "json", "formatversion": 2}
        )
        data = _http_get_json(f"{self.endpoint}?{params}", self.timeout_seconds, {"User-Agent": "research-agent/1.0"})
        records = data.get("query", {}).get("search", []) if isinstance(data, Mapping) else []
        results: list[Source] = []
        seen: set[str] = set()
        for record in records if isinstance(records, list) else []:
            if not isinstance(record, Mapping):
                continue
            title = _clean_text(record.get("title"))
            if not title:
                continue
            raw_url = record.get("url") or record.get("fullurl") or record.get("canonicalurl")
            url = str(raw_url or f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'), safe='()_:-')}")
            key = _canonical_url(url)
            if key in seen:
                continue
            seen.add(key)
            results.append(
                _make_source(
                    self.source_id,
                    title,
                    url,
                    record.get("snippet", ""),
                    kind="encyclopedia",
                    credibility="medium",
                )
            )
            if len(results) >= _limit(limit):
                break
        return results


@dataclass(slots=True)
class OpenAlexProvider:
    timeout_seconds: float = 10.0
    endpoint: str = "https://api.openalex.org/works"
    source_id: str = OPENALEX_SOURCE_ID

    def search(self, query: str, *, limit: int = 5) -> list[Source]:
        # OpenAlex treats question marks and asterisks as wildcards in stemmed
        # search and rejects them. Natural-language research questions often
        # contain a trailing question mark, so remove wildcard punctuation at
        # this provider boundary while preserving the semantic query text.
        safe_query = " ".join(str(query).replace("?", " ").replace("*", " ").split())
        params = urlencode({"search": safe_query, "per-page": _limit(limit)})
        data = _http_get_json(f"{self.endpoint}?{params}", self.timeout_seconds, {"User-Agent": "research-agent/1.0"})
        records = data.get("results", []) if isinstance(data, Mapping) else []
        results: list[Source] = []
        seen: set[str] = set()
        for record in records if isinstance(records, list) else []:
            if not isinstance(record, Mapping):
                continue
            title = _clean_text(record.get("title"))
            location = record.get("primary_location") if isinstance(record.get("primary_location"), Mapping) else {}
            landing = location.get("landing_page_url") or record.get("landing_page_url")
            url = landing or record.get("doi") or record.get("id")
            if not title or not url:
                continue
            url = str(url)
            if url.startswith("doi:"):
                url = f"https://doi.org/{url[4:]}"
            elif url.startswith("10."):
                url = f"https://doi.org/{url}"
            if not url.startswith(("http://", "https://")):
                continue
            key = _canonical_url(url)
            if key in seen:
                continue
            seen.add(key)
            abstract = record.get("abstract_inverted_index")
            snippet = _abstract_from_inverted_index(abstract)
            authors = [
                _read_value(_read_value(author, "author", {}), "display_name", "")
                for author in (record.get("authorships") or [])[:5]
            ]
            results.append(
                _make_source(
                    self.source_id,
                    title,
                    url,
                    snippet,
                    kind="academic",
                    published_at=record.get("publication_year"),
                    authors=authors,
                    credibility="high",
                )
            )
            if len(results) >= _limit(limit):
                break
        return results


def _abstract_from_inverted_index(value: Any) -> str:
    if not isinstance(value, Mapping):
        return _clean_text(value)
    words: list[tuple[int, str]] = []
    for word, positions in value.items():
        if not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, int):
                words.append((position, str(word)))
    return " ".join(word for _, word in sorted(words))


def _read_value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    try:
        return getattr(value, name)
    except (AttributeError, TypeError):
        return default


def _walk_values(value: Any, seen: set[int] | None = None) -> Iterable[Any]:
    """Walk SDK objects and dictionaries without assuming a concrete SDK version."""

    visited = set() if seen is None else seen
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return
    identity = id(value)
    if identity in visited:
        return
    visited.add(identity)
    yield value
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _walk_values(child, visited)
        return
    if isinstance(value, (list, tuple, set)):
        for child in value:
            yield from _walk_values(child, visited)
        return
    try:
        attributes = vars(value)
    except TypeError:
        attributes = {}
    for child in attributes.values():
        yield from _walk_values(child, visited)


def _output_text(response: Any) -> str:
    """Read output text from SDK responses and small dictionary fixtures."""

    direct = _read_value(response, "output_text", "")
    if isinstance(direct, str) and direct:
        return direct
    fragments: list[str] = []
    for node in _walk_values(response):
        if _read_value(node, "type") != "output_text":
            continue
        text = _read_value(node, "text", "")
        if isinstance(text, str) and text:
            fragments.append(text)
    return "".join(fragments)


def _nearby_citation_text(text: str, start_index: Any, end_index: Any) -> str:
    """Return the sentence surrounding an annotation, never the whole answer."""

    if not text or not isinstance(start_index, int) or not isinstance(end_index, int):
        return ""
    if start_index < 0 or end_index < start_index or start_index > len(text):
        return ""
    end_index = min(end_index, len(text))
    boundary_before = max(
        (text.rfind(mark, 0, start_index) for mark in ".!?\n"),
        default=-1,
    )
    boundary_after_candidates = [text.find(mark, end_index) for mark in ".!?\n"]
    boundary_after_candidates = [position for position in boundary_after_candidates if position >= 0]
    boundary_after = min(boundary_after_candidates, default=min(len(text), end_index + 320))
    snippet = text[boundary_before + 1 : boundary_after + 1]
    return _clean_text(snippet)[:1_200]


def _citation_field(node: Any, *names: str) -> Any:
    for name in names:
        value = _read_value(node, name)
        if value not in (None, "", [], {}):
            return value
    return None


def _citation_metadata(node: Any) -> dict[str, Any]:
    """Keep useful source metadata while avoiding response-wide payloads."""

    metadata: dict[str, Any] = {}
    raw_metadata = _citation_field(node, "metadata", "meta")
    if isinstance(raw_metadata, Mapping):
        metadata.update({str(key): value for key, value in raw_metadata.items()})
    for key in ("publisher", "domain", "source_type", "kind", "credibility", "published_at", "date"):
        value = _read_value(node, key)
        if value not in (None, ""):
            metadata.setdefault(key, value)
    return metadata


def _merge_citation(target: dict[str, Any], candidate: dict[str, Any]) -> None:
    """Merge an annotation and hosted source record for the same URL."""

    for key, value in candidate.items():
        if key == "metadata":
            if isinstance(value, Mapping):
                target.setdefault("metadata", {}).update(value)
            continue
        if value not in (None, "", [], {}):
            # Hosted records commonly contain a better source-specific snippet
            # than the surrounding output-text context.
            if key == "snippet" and target.get("snippet"):
                target["snippet"] = value if candidate.get("snippet_explicit") else target["snippet"]
            elif not target.get(key):
                target[key] = value


def extract_url_citations(response: Any) -> list[dict[str, Any]]:
    """Extract source-specific URL citations from SDK objects or dict fixtures.

    URL annotations contribute offsets and nearby output text. Hosted source
    records contribute their own snippets and metadata. Records are merged by
    canonical URL, preserving first-seen order and never using the complete
    response text as every source's evidence.
    """

    text = _output_text(response)
    citations: list[dict[str, Any]] = []
    by_url: dict[str, dict[str, Any]] = {}
    for node in _walk_values(response):
        node_type = _read_value(node, "type")
        url = _read_value(node, "url")
        title = _clean_text(_read_value(node, "title", _read_value(node, "name", "")))
        is_annotation = node_type == "url_citation"
        if (
            not isinstance(url, str)
            or not url.startswith(("http://", "https://"))
            or (not is_annotation and not title)
        ):
            continue
        start_index = _citation_field(node, "start_index", "start")
        end_index = _citation_field(node, "end_index", "end")
        explicit_snippet = _citation_field(node, "snippet", "evidence", "evidence_text", "description", "summary")
        if explicit_snippet is None and not is_annotation:
            explicit_snippet = _citation_field(node, "text", "content")
        nearby = _nearby_citation_text(text, start_index, end_index)
        snippet = _clean_text(explicit_snippet) or nearby
        candidate = {
            "url": url,
            "title": title,
            "snippet": snippet,
            "evidence": snippet,
            "start_index": start_index if isinstance(start_index, int) else None,
            "end_index": end_index if isinstance(end_index, int) else None,
            "metadata": _citation_metadata(node),
            "snippet_explicit": bool(_clean_text(explicit_snippet)),
            "source_type": node_type,
        }
        key = _canonical_url(url)
        if key not in by_url:
            by_url[key] = candidate
            citations.append(candidate)
        else:
            _merge_citation(by_url[key], candidate)
            by_url[key]["evidence"] = by_url[key].get("snippet", "")
    return citations


@dataclass(slots=True)
class OpenAIWebSearchProvider:
    """Optional provider backed by the Responses API built-in web search tool."""

    api_key: str | None = None
    model: str = "gpt-5.4-mini"
    client: Any = None
    source_id: str = OPENAI_WEB_SOURCE_ID

    @property
    def available(self) -> bool:
        return self.client is not None or bool(self.api_key)

    def _client(self) -> Any:
        if self.client is not None:
            return self.client
        if not self.api_key:
            return None
        from openai import OpenAI  # type: ignore[import-not-found]

        self.client = OpenAI(api_key=self.api_key)
        return self.client

    def search(self, query: str, *, limit: int = 5) -> list[Source]:
        client = self._client()
        if client is None:
            return []
        response = client.responses.create(
            model=self.model,
            tools=[{"type": "web_search"}],
            input=str(query),
            include=["web_search_call.action.sources"],
            max_output_tokens=1_200,
        )
        citations = extract_url_citations(response)
        results: list[Source] = []
        for citation in citations[: _limit(limit)]:
            metadata = citation.get("metadata") if isinstance(citation.get("metadata"), Mapping) else {}
            authors = metadata.get("authors", metadata.get("author", []))
            if isinstance(authors, str):
                authors = [authors]
            results.append(
                _make_source(
                    self.source_id,
                    citation.get("title"),
                    citation.get("url"),
                    citation.get("snippet", citation.get("evidence", "")),
                    kind=str(metadata.get("kind", metadata.get("source_type", "web"))),
                    published_at=metadata.get("published_at", metadata.get("date")),
                    authors=authors if isinstance(authors, Sequence) else [],
                    credibility=str(metadata.get("credibility", "unknown")),
                    evidence=citation.get("evidence", citation.get("snippet", "")),
                    start_index=citation.get("start_index"),
                    end_index=citation.get("end_index"),
                    metadata=metadata,
                )
            )
        return results


@dataclass(slots=True, init=False)
class FixtureProvider:
    """Deterministic provider for tests, demos, and offline development."""

    fixtures: Mapping[str, Sequence[Any]] | Sequence[Any] | None
    source_id: str

    def __init__(
        self,
        provider_name: str | Mapping[str, Sequence[Any]] | Sequence[Any] = FIXTURE_SOURCE_ID,
        sources: Mapping[str, Sequence[Any]] | Sequence[Any] | None = None,
        *,
        source_id: str | None = None,
        fixtures: Mapping[str, Sequence[Any]] | Sequence[Any] | None = None,
    ) -> None:
        """Accept both ``FixtureProvider(name, sources)`` and fixture-only forms."""

        if isinstance(provider_name, str):
            self.source_id = source_id or provider_name
            self.fixtures = fixtures if fixtures is not None else sources
        else:
            self.source_id = source_id or FIXTURE_SOURCE_ID
            self.fixtures = fixtures if fixtures is not None else provider_name

    def search(self, query: str, *, limit: int = 5) -> list[Source]:
        fixtures = self.fixtures
        if isinstance(fixtures, Mapping):
            raw = fixtures.get(str(query), fixtures.get(str(query).casefold(), fixtures.get("*", ())))
        else:
            raw = fixtures or ()
        results: list[Source] = []
        for item in list(raw)[: _limit(limit)]:
            if isinstance(item, Source):
                results.append(item)
                continue
            if isinstance(item, Mapping):
                results.append(
                    _make_source(
                        self.source_id,
                        item.get("title"),
                        item.get("url"),
                        item.get("snippet", item.get("summary", "")),
                        kind=str(item.get("kind", "web")),
                        published_at=item.get("published_at"),
                        authors=item.get("authors"),
                        credibility=str(item.get("credibility", "unknown")),
                    )
                )
        return results


# Descriptive aliases keep the provider layer easy to discover without
# changing the stable provider IDs used in returned records.
WikipediaSourceProvider = WikipediaProvider
OpenAlexSourceProvider = OpenAlexProvider
OpenAIWebSearchSourceProvider = OpenAIWebSearchProvider
FixtureSourceProvider = FixtureProvider
_extract_url_citations = extract_url_citations


def providers_from_settings(settings: Any) -> list[SourceProvider]:
    """Create the public providers enabled by a Settings-like object."""

    providers: list[SourceProvider] = []
    timeout = getattr(settings, "request_timeout_seconds", getattr(settings, "timeout_seconds", 10.0))
    if getattr(settings, "wikipedia_enabled", True):
        providers.append(WikipediaProvider(timeout_seconds=timeout, endpoint=getattr(settings, "wikipedia_endpoint", WikipediaProvider.endpoint)))
    if getattr(settings, "openalex_enabled", True):
        providers.append(OpenAlexProvider(timeout_seconds=timeout, endpoint=getattr(settings, "openalex_endpoint", OpenAlexProvider.endpoint)))
    if getattr(settings, "openai_web_enabled", False) and getattr(settings, "openai_api_key", None):
        providers.append(
            OpenAIWebSearchProvider(
                api_key=settings.openai_api_key,
                model=getattr(settings, "openai_model", "gpt-5.4-mini"),
            )
        )
    return providers


def _invoke_retry(function: Any, retry_policy: Any) -> Any:
    try:
        parameters = inspect.signature(retry_call).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "retry_policy" in parameters:
        return retry_call(function, retry_policy=retry_policy)
    if "policy" in parameters and not isinstance(retry_policy, Mapping):
        return retry_call(function, policy=retry_policy)
    # Some retry helpers expose policy values as keyword arguments rather than
    # accepting a policy object.  Forward only recognized values.
    policy_values = dict(retry_policy) if isinstance(retry_policy, Mapping) else {}
    if not policy_values:
        try:
            policy_values = {
                name: getattr(retry_policy, name)
                for name in dir(retry_policy)
                if not name.startswith("_") and not callable(getattr(retry_policy, name, None))
            }
        except Exception:
            policy_values = {}
    supported = {
        name
        for name, parameter in parameters.items()
        if name != "function"
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    forwarded = {name: value for name, value in policy_values.items() if name in supported}
    return retry_call(function, **forwarded)


@dataclass(slots=True)
class MultiSourceSearchTool:
    """Fan out a query, isolate provider failures, retry, and deduplicate URLs."""

    providers: Sequence[SourceProvider]
    retry_policy: Any = None
    max_workers: int | None = None

    def _search_provider(self, provider: SourceProvider, query: str, limit: int) -> Sequence[Source]:
        def call() -> Sequence[Source]:
            try:
                parameters = inspect.signature(provider.search).parameters
            except (TypeError, ValueError):
                parameters = {}
            if "limit" in parameters or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
            ):
                return provider.search(query, limit=limit)
            if "max_results" in parameters:
                return provider.search(query, max_results=limit)  # type: ignore[call-arg]
            return provider.search(query)  # type: ignore[call-arg]

        if self.retry_policy is None:
            return call()
        return _invoke_retry(call, self.retry_policy)

    def search(self, query: str, max_results: int = 5, *, limit: int | None = None) -> SearchBundle:
        query = str(query)
        result_limit = _limit(max_results if limit is None else limit)
        providers = list(self.providers)
        statuses: list[ProviderStatus | None] = [None] * len(providers)
        provider_results: list[Sequence[Source]] = [()] * len(providers)

        worker_count = max(1, min(len(providers), self.max_workers or len(providers))) if providers else 0
        if worker_count:
            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="research-source") as executor:
                pending = {
                    executor.submit(self._search_provider, provider, query, result_limit): index
                    for index, provider in enumerate(providers)
                }
                for future in as_completed(pending):
                    index = pending[future]
                    provider = providers[index]
                    provider_id = _provider_id(provider)
                    try:
                        value = future.result()
                        if isinstance(value, Mapping):
                            value = value.get("sources", value.get("results", ()))
                        results = list(value or ())
                        provider_results[index] = results
                        statuses[index] = _make_status(provider_id, True, len(results))
                    except Exception as exc:  # one provider must not erase other providers' results
                        statuses[index] = _make_status(provider_id, False, 0, f"{type(exc).__name__}: {exc}")

        sources: list[Source] = []
        seen_urls: set[str] = set()
        for offset in range(result_limit):
            for results in provider_results:
                if offset >= len(results):
                    continue
                source = results[offset]
                url = _read_value(source, "url", "")
                key = _canonical_url(str(url))
                if not key or key in seen_urls:
                    continue
                seen_urls.add(key)
                sources.append(source)
                if len(sources) >= result_limit:
                    break
            if len(sources) >= result_limit:
                break
        final_statuses = [status for status in statuses if status is not None]
        return _make_bundle(query, sources, final_statuses)

    def __call__(self, query: str, max_results: int = 5, *, limit: int | None = None) -> SearchBundle:
        return self.search(query, max_results, limit=limit)
