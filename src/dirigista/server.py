"""MCP surface: registers the classification tools and delegates to `classifiers`.

Deliberately thin. It validates, opens a client, delegates and serializes; it
holds no classification logic of its own.
"""

from mcp.server.mcpserver import MCPServer

from . import classifiers
from .client import inference_client
from .models import CategorizeResult, ScoreResult, VerifyResult

mcp = MCPServer(
    name="dirigista",
    instructions=(
        "Delegate any classification of text to these tools instead of judging "
        "it yourself: routing, triage, labelling, moderation, prioritising, "
        "sentiment, intent, relevance and fact checks. One item per call.\n"
        "Pick by the answer you need: verify (true or false), categorize (one "
        "of your options), score (a level on your scale).\n"
        "Answers are probabilities, never explanations."
    ),
)


@mcp.tool()
async def verify(
    statement: str,
    context: str | None = None,
    false_below: float = 0.2,
    true_above: float = 0.8,
) -> VerifyResult:
    """Judge whether a claim is true or false.

    Use for fact checks and any yes/no judgment about text. Contested claims
    come back 'uncertain' rather than forced either way. Pass `context` to
    judge against a specific source instead of general knowledge. Move the
    thresholds when a false positive and a false negative cost you differently.
    """
    async with inference_client() as client:
        return await classifiers.verify(
            statement,
            context,
            client=client,
            false_below=false_below,
            true_above=true_above,
        )


@mcp.tool()
async def categorize(statement: str, categories: list[str]) -> CategorizeResult:
    """Pick the one category that best fits a piece of text.

    Use for routing, triage, labelling, intent and any "which of these is it"
    decision. Any taxonomy works: you supply the categories. Descriptive names
    beat terse codes, and an option you do not offer cannot be chosen — add
    "none of the above" if your list may not cover everything.
    """
    async with inference_client() as client:
        return await classifiers.categorize(statement, categories, client=client)


@mcp.tool()
async def score(statement: str, criteria: list[str]) -> ScoreResult:
    """Rate a piece of text on a scale you define.

    Use for severity, urgency, priority, sentiment strength, quality, relevance
    and any "how much" judgment. Give `criteria` lowest to highest, each level
    described in words, e.g. ["Calm, just stating facts", "Frustrated but
    civil", "Very angry, strong language"]. Concrete descriptions rate far more
    reliably than bare "low"/"medium"/"high".
    """
    async with inference_client() as client:
        return await classifiers.score(statement, criteria, client=client)
