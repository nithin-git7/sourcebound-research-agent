# Release checklist

## Local gates

Run these commands from the repository root:

```powershell
python -m unittest discover -v
python -m research_agent evaluate
python -m compileall -q research_agent tests
python -m pip wheel . --no-deps --no-build-isolation --wheel-dir dist
```

For the optional web surface:

```powershell
python -m pip install -e ".[dev,web]"
uvicorn research_agent.api:create_app --factory --host 127.0.0.1 --port 8000
```

Check `GET /health`, `GET /report`, and `GET /api/report`, then serve
`portfolio/` over HTTP and replay every trace stage, including a citation jump.

## GitHub

1. Create or select the destination repository.
2. Add its remote and push the intended branch.
3. Confirm the workflow passes on Python 3.11 and 3.12.
4. Enable the repository's static hosting with `portfolio/` as the published
   directory, or copy that directory into the portfolio site's normal static
   content path.

## Live release checks

- Run one live research request with an environment-only `OPENAI_API_KEY`.
- Confirm provider failures remain visible and no credentials appear in output.
- Build and smoke-test the Docker image if Docker is available.
- Review the generated report and citations with a human before presenting it
  as current research.

This repository contains no configured Git remote or deployment target, so the
GitHub push and portfolio publication steps remain explicit operator actions.
