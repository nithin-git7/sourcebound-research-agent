# Sourcebound Research Agent

Sourcebound is a small, production-shaped research agent that searches independent source providers, compares their positions, and returns a citation-grounded report.

It is deliberately designed as an interview-ready system rather than a single prompt:

- A model calls a typed search_sources function tool.
- The tool fans out to independent providers in parallel and keeps provider-level failures visible.
- The model receives a strict JSON Schema response contract for the report draft.
- Exponential backoff handles transient model and source failures.
- A bounded retrieval planner decomposes questions into focused intents, caps
  query/result budgets, prefers fresh and credible evidence, and reranks for
  provider diversity with an explicit stop decision.
- Hosted-search citations preserve source-specific excerpts, response offsets,
  and metadata instead of attaching the entire model response to every URL.
- A grounding validator rejects citations that were not returned by the search tool.
- A claim verifier matches cited claims to returned evidence and exposes support gaps.
- An evaluation suite measures retrieval coverage, claim support, answer completeness, and report structure.
- A security boundary validates source URLs, bounds retrieved text, detects common prompt-injection patterns, and marks evidence as untrusted before it reaches the model.
- A static portfolio viewer makes the question-to-verification trace inspectable, while an optional FastAPI adapter exposes the same validated report over HTTP.

## Quick start

The project is runnable without an API key in deterministic fixture mode:

~~~powershell
python -m pip install -e .
python -m research_agent research "What are the trade-offs of retrieval-augmented generation?" --offline --format markdown
python -m research_agent evaluate
python -m unittest discover -v
~~~

For live research, create a virtual environment, install the optional OpenAI dependency, and set OPENAI_API_KEY:

~~~powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[live,dev]"
$env:OPENAI_API_KEY = "your-key"
python -m research_agent research "How are battery recycling policies evolving?" --format markdown
~~~

If no key is present, the CLI automatically falls back to offline mode and tells you why. Offline mode is a reproducible demo and evaluation path, not a substitute for current web research.

## Portfolio and local API

The self-contained portfolio case study lives in `portfolio/` and can be served
from any static host. Serve it over HTTP so the viewer can load its deterministic
sample report:

~~~powershell
python -m http.server 8765 --directory portfolio
~~~

The optional API adapter serves a validated report for a portfolio host or a
local integration test:

~~~powershell
python -m pip install -e ".[web]"
uvicorn research_agent.api:create_app --factory --host 127.0.0.1 --port 8000
~~~

It exposes `/health`, `/report`, and `/api/report`. Set
`SOURCEBOUND_REPORT_PATH` to serve a specific JSON report; otherwise the sample
fixture is used when the source tree is present, with an offline fallback after
installation. The adapter is intentionally optional and the core package does
not import FastAPI at module import time.

Build the same web slice in a container with:

~~~powershell
docker build -t sourcebound-research-agent .
docker run --rm -p 8000:8000 sourcebound-research-agent
~~~

## Architecture

~~~text
question
   |
   v
bounded retrieval planner
   |-- focused intents and freshness requirements
   |-- query/result budgets and stop criteria
   +-- relevance + credibility + recency + diversity reranking
   |
   v
Responses API + strict report schema
   |  function call: search_sources(query, max_results)
   v
MultiSourceSearchTool
   |-- Wikipedia search
   |-- OpenAlex academic search
   +-- OpenAI hosted web search (live mode)
   |
   v
source bundle with stable IDs and source-specific evidence spans
   |
   v
structured report draft
   |
   v
ID grounding + claim evidence verification + Markdown/JSON renderer
~~~

The live path uses the OpenAI Responses API tools interface for the custom function and text.format with json_schema for the structured report contract. The hosted web-search adapter is optional; Wikipedia and OpenAlex provide independent public-source coverage without another API key.

## Report contract

The model is only allowed to cite source IDs returned by search_sources. A final report contains:

- an executive summary;
- key findings with confidence and citation IDs;
- comparison points with source-by-source positions;
- limitations;
- the exact source records used;
- an evidence audit with citation coverage, provider diversity, unresolved citations, and warnings;
- per-claim evidence checks with supported, partial, unsupported, or contradicted verdicts;
- retrieval-plan traces with planned queries, coverage, and stop reasons;
- tool-call traces for observability.

The app owns source metadata and URLs. The model owns synthesis. This separation reduces the risk of fabricated links and makes the provenance check deterministic.

Claim verification is intentionally conservative and provider-agnostic. The default
implementation uses lexical overlap and explicit polarity cues to surface matched
passages; it is not semantic entailment. A future LLM judge can implement the same
verifier protocol without changing the report contract.

## Evaluation

The built-in evaluation suite uses the same agent loop with a deterministic model and curated fixture providers. It is versioned in `research_agent/data/research_benchmark.json` and checks:

~~~text
score = 35% citation coverage
      + 25% citation grounding
      + 20% source diversity
      + 20% comparison quality

benchmark metrics additionally report:
  retrieval recall over expected concepts
  claim-support proxy over cited findings
  answer completeness over expected concepts
~~~

Run it with:

~~~powershell
python -m research_agent evaluate
~~~

The suite is intentionally cheap and repeatable. Its metrics are deterministic
offline proxies, not live-web quality measurements or a substitute for human
review. A next step for a deployed system would be to add a human-labeled set and
run live-vs-fixture regression checks in CI.

The repository workflow runs the unittest suite, Python compilation, and a
wheel build on Python 3.11 and 3.12. Docker-image construction and live-provider
execution remain release-time checks because they require external tooling or
credentials.

## Resume-ready summary

Built a citation-grounded research agent in Python using the OpenAI Responses API, custom function tools, strict JSON Schema structured outputs, parallel multi-source retrieval, exponential-backoff retries, deterministic claim-to-evidence verification, and a versioned benchmark harness for retrieval coverage, claim support, answer completeness, and cross-source comparison quality.

Extended it with bounded query planning, explainable source reranking, explicit retrieval stop criteria, and source-specific hosted-search provenance with evidence offsets and metadata.

The portfolio layer adds a replayable trace viewer, an optional HTTP report
adapter, a security policy, a CI workflow, and a truthful container entrypoint.

## API references

- [Create a model response](https://developers.openai.com/api/reference/cli/resources/responses/methods/create)
- [OpenAI API quickstart](https://platform.openai.com/docs/quickstart/make-your-first-api-request)
