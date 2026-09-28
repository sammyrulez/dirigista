"""How the server finds its API key.

An MCP client starts the server as a subprocess with a minimal environment —
only HOME, LOGNAME, PATH, SHELL, TERM and USER are forwarded. Anything the
server needs must therefore come from its own working directory, not from the
environment it was launched with.
"""

import os

import pytest

from dirigista import client


def test_the_api_key_is_read_from_the_environment_when_it_is_there(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "key-from-env")

    assert client.resolve_api_key() == "key-from-env"


def test_the_api_key_falls_back_to_a_dotenv_file(tmp_path, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    (tmp_path / ".env").write_text("TYPESAFE_API_KEY=key-from-dotenv\n")
    monkeypatch.chdir(tmp_path)

    assert client.resolve_api_key() == "key-from-dotenv"


def test_a_missing_key_names_the_variable_instead_of_failing_deep_in_the_sdk(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        client.resolve_api_key()


def test_a_dotenv_file_cannot_inject_anything_but_the_key(tmp_path, monkeypatch):
    """A .env is untrusted input: only the key may be taken from it.

    Loading a .env into the process environment would let any variable in it
    take effect — TYPESAFE_BASE_URL redirects every request (API key included)
    to a host of the file's choosing, and httpx honours HTTPS_PROXY and
    SSL_CERT_FILE from the environment too.
    """
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_BASE_URL", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    (tmp_path / ".env").write_text(
        "TYPESAFE_API_KEY=key-from-dotenv\n"
        "TYPESAFE_BASE_URL=https://attacker.example\n"
        "HTTPS_PROXY=http://attacker.example:8080\n"
    )
    monkeypatch.chdir(tmp_path)

    assert client.resolve_api_key() == "key-from-dotenv"
    assert "TYPESAFE_BASE_URL" not in os.environ
    assert "HTTPS_PROXY" not in os.environ


def test_a_dotenv_above_the_working_directory_is_not_used(tmp_path, monkeypatch):
    """Searching up the tree would read a file the user never pointed us at."""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    (tmp_path / ".env").write_text("TYPESAFE_API_KEY=key-from-a-parent\n")
    workdir = tmp_path / "project"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    with pytest.raises(RuntimeError):
        client.resolve_api_key()
