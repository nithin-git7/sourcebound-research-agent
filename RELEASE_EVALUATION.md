# Sourcebound 0.2 release evaluation

Evaluated on 2026-09-03. This is a transparent project-readiness heuristic,
not an external certification or hiring score.

## Result: 92 / 100

| Area | Score | Evidence |
| --- | ---: | --- |
| Agent architecture | 19 / 20 | Bounded planning, tool calling, strict outputs, provider fan-out, and application-owned provenance |
| Citation integrity | 18 / 20 | Unknown IDs fail validation; lexical checks are explicit; optional semantic checks are isolated |
| Resilience and observability | 15 / 15 | Classified retries, provider isolation, bounded jobs, lifecycle states, and thread-safe telemetry |
| Evaluation quality | 16 / 20 | 25 deterministic cases across five domains with regression thresholds and proxy disclosure |
| Product experience | 14 / 15 | Live two-provider public demo, deterministic replay, six-stage trace, responsive and accessible states |
| Release hygiene | 10 / 10 | 77 tests, correctness lint, Python 3.11 and 3.12 CI, wheel build, MIT license, changelog, and verified Pages release |

## Verified gates

- 77 tests passed with 35 parameterized subtests.
- The 25-case recorded-fixture benchmark passed its declared regression thresholds.
- Python compilation, Ruff correctness checks, and the 0.2.0 wheel build passed.
- The local FastAPI smoke test completed a research job and returned telemetry.
- The deployed browser demo retrieved three Wikipedia and three OpenAlex records.
- Desktop and 390 px mobile checks showed no horizontal overflow.
- GitHub Actions and GitHub Pages completed successfully for the release commits.

## Honest limitations

- The public GitHub Pages demo performs live retrieval and extractive citation display, not model synthesis.
- Full synthesis and semantic verification require a separately hosted Python API and an OpenAI API key.
- The benchmark uses curated recorded fixtures and lexical or structural proxies, not human labels or live-web judgments.
- Token and cost telemetry primitives exist, but live model-usage accounting still needs provider usage fields wired into the agent runtime.

## Best next investment

Deploy the Python API behind authentication, connect response usage to cost
telemetry, and add a small human-labeled evaluation set. Those changes would
raise confidence more than adding more interface polish.
