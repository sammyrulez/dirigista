"""Construction of the client that reaches the inference backend.

Kept apart from `classifiers` so the classification logic can be tested against
a stand-in client without touching the network or an API key.
"""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import dotenv_values
from typesafe_sdk import AsyncTypeSafeClient

API_KEY_ENV = "TYPESAFE_API_KEY"
OTEL_TOKEN_ENV = "OTEL_BEARER_TOKEN"


def _from_env_or_dotenv(name: str) -> str | None:
    """One credential from the environment, else from ./.env — read, never loaded."""
    value = os.environ.get(name)
    if value:
        return value

    # Read the file, do not load it. A .env is untrusted input: anything it
    # defines would otherwise take effect on this process, and the SDK resolves
    # its endpoint from TYPESAFE_BASE_URL while httpx honours HTTPS_PROXY and
    # SSL_CERT_FILE. A single planted line would send this key, and every
    # statement classified, to a host of the file's choosing. Only the named
    # value is taken, and only from the directory the client was told to run
    # in — never from a parent the user never pointed us at.
    dotenv = Path.cwd() / ".env"
    if dotenv.is_file():
        return (dotenv_values(dotenv).get(name) or "").strip() or None
    return None


def resolve_api_key() -> str:
    """Find the API key, falling back to a .env file in the working directory.

    The fallback is not a convenience. An MCP client starts this server as a
    subprocess with a deliberately minimal environment — HOME, LOGNAME, PATH,
    SHELL, TERM, USER and nothing else — so a key exported in the user's shell
    never reaches us. Reading it from the project directory is what makes the
    server work when launched the way it is meant to be launched.

    The environment is trusted; the file is not. Only the key is read from it,
    and nothing in it is ever loaded into this process.
    """
    key = _from_env_or_dotenv(API_KEY_ENV)
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


def resolve_otel_token() -> str | None:
    """The bearer token for the trace collector, if one is configured.

    Resolved like the API key, for the same reason: it is a secret, and the
    stripped environment of an MCP subprocess may only reach it through .env.
    Taking it from the file is safe where taking the endpoint would not be —
    the endpoint comes from the environment alone, so the file cannot choose
    where the token is sent.
    """
    return _from_env_or_dotenv(OTEL_TOKEN_ENV)
