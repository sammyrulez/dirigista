"""Live integration with the inference backend. Skipped without an API key.

These tests spend money and need the network, so they are opt-in:

    uv run pytest -m live
"""

import pytest

from dirigista import classifiers, client
from dirigista.client import inference_client


def _has_api_key() -> bool:
    """Ask the server's own resolver, so the skip matches what the code does."""
    try:
        client.resolve_api_key()
    except RuntimeError:
        return False
    return True


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not _has_api_key(), reason="no TYPESAFE_API_KEY available"),
]


async def test_a_plainly_true_claim_is_verified():
    async with inference_client() as client:
        result = await classifiers.verify(
            "Paris is the capital of France.", client=client
        )

    assert result.verdict == "true"


async def test_a_genuinely_contested_claim_is_left_undecided():
    """The three-valued verdict earns its keep here.

    "Water is wet" is a real dispute — water wets other things, but whether it
    is itself wet is contested — and the model puts it near 0.5 rather than
    picking a side. A boolean tool would have to invent an answer.
    """
    async with inference_client() as client:
        result = await classifiers.verify("Water is wet.", client=client)

    assert result.verdict == "uncertain"


async def test_a_plainly_false_claim_is_rejected():
    async with inference_client() as client:
        result = await classifiers.verify("The sun orbits the Earth.", client=client)

    assert result.verdict == "false"


async def test_the_fitting_category_is_picked():
    async with inference_client() as client:
        result = await classifiers.categorize(
            "My package arrived smashed and the contents are broken.",
            ["shipping damage", "billing question", "password reset"],
            client=client,
        )

    assert result.category == "shipping damage"


async def test_an_angry_message_lands_high_on_the_tone_scale():
    async with inference_client() as client:
        result = await classifiers.score(
            "This is the fourth time I am writing and NOBODY has bothered to reply.",
            [
                "Calm, just stating facts",
                "Frustrated but civil",
                "Very angry, strong language",
            ],
            client=client,
        )

    assert result.level >= 2


async def test_a_tool_call_survives_the_stripped_environment_of_an_mcp_client():
    """The regression that unit tests could not see.

    A client launches the server with a minimal environment, so a key that only
    exists in the parent shell never reaches it. Every tool failed this way
    while all unit tests were green.
    """
    import sys

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=sys.executable, args=["-m", "dirigista"])

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "verify", {"statement": "Rome is the capital of Italy."}
            )

    assert not result.is_error
    assert result.structured_content["verdict"] == "true"
