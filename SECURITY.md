# Security

Sourcebound treats retrieved pages, titles, snippets, and hosted-search
annotations as untrusted input. The application validates source URLs, bounds
evidence sizes, detects common prompt-injection patterns, and keeps source
content separate from agent instructions.

## Safe operation

- Never commit `.env` files, API keys, or live research output containing
  sensitive information.
- Keep `OPENAI_API_KEY` in the environment; do not put it in reports,
  screenshots, fixtures, or issue descriptions.
- Review citations and the claim-verification section before treating a report
  as authoritative.
- Use offline fixtures for reproducible demos and tests. Live reports can be
  stale, incomplete, or affected by provider failures.

## Reporting a vulnerability

Please avoid publishing exploit details in a public issue. Contact the project
maintainer privately with a reproducible description, affected file or route,
impact, and a minimal proof of concept. Do not include secrets or personal
data in the report.
