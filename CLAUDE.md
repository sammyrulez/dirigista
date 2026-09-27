# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```sh
uv run pytest                          # unit + contract tests; no network, no cost
uv run pytest -m live                  # also hits the real API; costs money
uv run pytest tests/test_client.py     # one file
uv run pytest -k dotenv                # one test by name
uv run python -m dirigista             # run the server (stdio)
```

Live tests are deselected by default via `addopts` in `pyproject.toml`. They self-skip when no API
key resolves, using the server's own resolver so the skip matches real behaviour.

If imports fail with `No module named 'dirigista'` after adding files, the editable install is
stale: `uv sync --reinstall-package dirigista`.

## Architecture

An MCP server exposing three classification tools. One direction of dependency:

```
server.py  → classifiers.py → client.py
(MCP only)   (logic only)     (credentials only)
             models.py (contracts, shared)
```

- **`server.py`** registers the tools and holds no logic: validate, open a client, delegate,
  serialize. Tool docstrings and the server `instructions` are consumed by the calling model, so
  they are product surface, not comments.
- **`classifiers.py`** knows nothing about MCP and takes its client as an argument. That injection
  is what makes the conversion logic testable without the network — `tests/conftest.py` supplies a
  fake client that records requests and replays canned answers.
- **`client.py`** resolves credentials and nothing else.

## Constraints that are easy to break

**Tool definitions must stay under 1000 tokens total.** They are re-sent on every request. Measure
before and after any wording change — schema field `description`s and Pydantic class docstrings both
end up in the JSON Schema, so they cost context exactly like tool descriptions do. Current budget:
~993 (instructions 77, verify 311, categorize 298, score 307).

**No tool returns a rationale.** The backend returns calibrated probabilities, never prose. A
`reasoning` field could only be invented after the fact, so trust is carried by the numbers.

**The backend vendor is named only in configuration instructions** (README requirements, the MCP
`env` snippet). Keep it out of tool descriptions, server instructions, model field descriptions and
code comments — those say "the inference backend" or "the model". SDK imports and
`TYPESAFE_API_KEY` are the unavoidable exceptions.

**An MCP client strips the environment.** The server subprocess receives only `HOME`, `LOGNAME`,
`PATH`, `SHELL`, `TERM`, `USER` — a key exported in the user's shell never arrives. `resolve_api_key`
therefore falls back to a `.env` found with `usecwd=True`, anchored to the working directory the
client was told to use. Unit tests cannot see a regression here; `test_live.py` covers it end to end
through a real client session.

**Level numbering differs between the contract and the backend.** `ScoreResult.level` is 1-based;
the backend's levels are 0-based. `classifiers.score` also returns the *most probable* level rather
than rounding the backend's probability-weighted mean, which would answer a different question.

## MCP SDK

This uses the v2 Python SDK: the class is `MCPServer` (not `FastMCP`), and attributes are snake_case
(`input_schema`, `output_schema`, `structured_content`). Most examples online show the v1 names.
