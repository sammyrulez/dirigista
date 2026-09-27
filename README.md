# dirigista

An [MCP](https://modelcontextprotocol.io) server that performs semantic classification tasks.

`dirigista` exposes three general-purpose classification tools to any MCP client, so an agent can
delegate a judgment about text instead of making it itself. Each tool takes a statement and returns
a typed judgment together with the probabilities behind it, so the caller can tell a confident
answer from a borderline one.

Between them the three cover routing, triage, labelling, filtering, moderation, prioritising,
ranking, sentiment, intent detection, relevance checks and fact checks — one item per call.

The judgments are calibrated probabilities rather than generated text, which has one consequence
worth stating plainly: **no tool returns a rationale.** An explanation would have to be invented
after the fact, and an invented rationale is worse than none. Trust is carried by the numbers.

## Tools

| Tool | Use it for | Input | Output |
| --- | --- | --- | --- |
| `verify` | fact checks, yes/no judgments | `statement`, `context?`, `false_below=0.2`, `true_above=0.8` | `verdict: "true" \| "false" \| "uncertain"`, `probability: float` |
| `categorize` | routing, triage, labelling | `statement`, `categories: list[str]` | `category: str`, `confidence: float`, `probabilities: dict[str, float]` |
| `score` | severity, urgency, intensity | `statement`, `criteria: list[str]` | `criterion: str`, `level: int`, `confidence: float` |

### `verify`

Decides whether a statement is true or false. The answer is a single probability that is both the
verdict and its certainty: near 1 a strong yes, near 0 a strong no, near 0.5 genuinely undecided.

The verdict is that probability read against two thresholds the caller can move (`false_below`,
`true_above`). It is deliberately three-valued: forcing a contested claim into true or false would
mean inventing an answer. Asked whether *"Water is wet"*, the model returns ≈0.51 — water wets other
things, but whether it is itself wet is a real dispute — and the tool reports `uncertain`.

The defaults 0.2 / 0.8 are a convention, not a fact about your domain. Move them when a false
positive and a false negative cost different amounts.

### `categorize`

Picks the single most appropriate category for a statement out of a caller-supplied list. The list is
part of the request, so the server carries no fixed taxonomy. The full distribution is returned
alongside the winner, and `confidence` reflects how concentrated it is: a flat spread means the
categories were not well separated for this input.

Include a catch-all such as `"none of the above"` when the list might not cover every input — the
model cannot choose an option you did not offer.

### `score`

Rates a statement against a caller-supplied list of criteria, treated as an **ordinal scale** ordered
from lowest to highest. The tool returns the level that best describes the statement, along with its
1-based position on the scale.

The underlying score is a probability-weighted mean and can land between levels (1.43 on a scale of
three). Since the tool promises a level, it reports the **most probable** one rather than rounding
that mean, which would answer a different question.

```json
{
  "statement": "I have asked for a refund three times and nobody has replied.",
  "criteria": [
    "Calm, just stating facts",
    "Frustrated but civil",
    "Very angry, strong language"
  ]
}
```

## Status

Working. All three tools classify through the inference backend, and the integration is covered end
to end: unit tests run against a stand-in client, and an opt-in suite hits the live API.

## Layout

```
src/dirigista/
  models.py          # Pydantic output models for the three tools
  classifiers.py     # builds the model questions, converts answers (no MCP knowledge)
  client.py          # constructs the inference client, resolves the API key
  server.py          # MCPServer instance, thin adapter over classifiers
  __main__.py        # stdio entrypoint
tests/
  test_classifiers.py # answer-to-result conversion, against a fake client
  test_client.py      # API key resolution
  test_server.py      # tool registration and schema contracts
  test_stdio.py       # end-to-end handshake with a real MCP client
  test_live.py        # opt-in, hits the real API
```

`server.py` holds no logic: it validates, opens a client, delegates and serializes. `classifiers.py`
knows nothing about MCP and takes its client as an argument, so the conversion logic is testable
without the network.

## Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)
- `mcp` >= 2 (the v2 SDK, where `FastMCP` was renamed `MCPServer`)
- Credentials for the inference backend. Classification runs on
  [TypeSafe](https://docs.typesafe.ai)'s Jev model, so you need a TypeSafe API key from
  [console.typesafe.ai](https://console.typesafe.ai/), in `TYPESAFE_API_KEY` or in a `.env` file in
  the directory the server runs from. The key stays server-side: it is never passed over MCP, so
  clients can neither supply nor read it.

## Running

```sh
uv run python -m dirigista
```

The server speaks MCP over stdio and is meant to be started as a subprocess by an MCP client.

**The key must reach the subprocess.** An MCP client launches the server with a deliberately minimal
environment — `HOME`, `LOGNAME`, `PATH`, `SHELL`, `TERM`, `USER` and nothing else — so a key exported
in your shell does *not* reach it. Either keep a `.env` in the project directory (found via
`--directory` below), or pass the key explicitly through the client's `env` setting:

```json
{
  "mcpServers": {
    "dirigista": {
      "command": "uv",
      "args": ["--directory", "/path/to/dirigista", "run", "python", "-m", "dirigista"],
      "env": { "TYPESAFE_API_KEY": "..." }
    }
  }
}
```

## Tests

```sh
uv run pytest           # unit and contract tests, no network, no cost
uv run pytest -m live   # also hits the real API; needs TYPESAFE_API_KEY
```

The live tests are deselected by default because they cost money and need the network.
