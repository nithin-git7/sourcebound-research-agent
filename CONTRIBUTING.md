# Contributing

Sourcebound is a small Python project with a deterministic offline path. Keep
changes focused, preserve the strict data contracts, and add regression tests
for behavior changes.

## Development

~~~powershell
python -m pip install -e ".[dev]"
python -m unittest discover -v
python -m research_agent evaluate
python -m compileall -q research_agent tests
~~~

To exercise the optional portfolio API locally, install the web extra and run
the factory-backed server:

~~~powershell
python -m pip install -e ".[dev,web]"
uvicorn research_agent.api:create_app --factory --reload
~~~

Live provider work is optional. Use the `live` extra and an environment-only
`OPENAI_API_KEY`; do not add credentials to fixtures or committed files.

## Pull requests

Explain the user-visible behavior, the contracts affected, and the evidence
used to validate the change. Include focused tests and note any live-network
or deployment checks that were not run.
