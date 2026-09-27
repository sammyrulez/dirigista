"""Construction of the client that reaches the inference backend.

Kept apart from `classifiers` so the classification logic can be tested against
a stand-in client without touching the network or an API key.
"""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from dotenv import find_dotenv, load_dotenv
from typesafe_sdk import AsyncTypeSafeClient

API_KEY_ENV = "TYPESAFE_API_KEY"


def resolve_api_key() -> str:
    """Find the API key, falling back to a .env file in the working directory.

    The fallback is not a convenience. An MCP client starts this server as a
    subprocess with a deliberately minimal environment — HOME, LOGNAME, PATH,
    SHELL, TERM, USER and nothing else — so a key exported in the user's shell
    never reaches us. Reading it from the project directory is what makes the
    server work when launched the way it is meant to be launched.
    """
    key = os.environ.get(API_KEY_ENV)
    if key:
        return key

    # Anchored to the working directory on purpose: an MCP client is told
    # which directory to run the server from, and that is where the project's
    # .env lives. Letting dotenv walk up from this module's own location would
    # find a different file depending on where the package is installed.
    dotenv = find_dotenv(usecwd=True)
    if dotenv:
        load_dotenv(dotenv)
    key = os.environ.get(API_KEY_ENV)
    if key:
        return key

    raise RuntimeError(
        f"{API_KEY_ENV} is not set. Put it in a .env file in the directory the "
        f"server runs from, or pass it through the MCP client's `env` setting."
    )


@asynccontextmanager
async def inference_client() -> AsyncIterator[AsyncTypeSafeClient]:
    """Open a client for the model that answers the classification questions.

    The key is resolved server-side and never travels over MCP, so callers of
    the tools can neither supply nor read it.
    """
    async with AsyncTypeSafeClient(api_key=resolve_api_key()) as client:
        yield client
