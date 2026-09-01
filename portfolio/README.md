# Sourcebound portfolio surface

This directory is a dependency-free static case study. It loads
`sample_report.json` and renders a six-stage research trace in `index.html`.

Preview it over HTTP from the repository root:

```powershell
python -m http.server 8765 --directory portfolio
```

Then open `http://127.0.0.1:8765/`. The page also includes loading, empty, and
error states so a hosted report failure is visible rather than silently hidden.

The optional Python adapter in `research_agent/api.py` serves a validated report
for hosts that need an HTTP endpoint. The static viewer itself does not require
Python, FastAPI, a database, or an API key.
