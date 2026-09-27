"""How the server finds its API key.

An MCP client starts the server as a subprocess with a minimal environment —
only HOME, LOGNAME, PATH, SHELL, TERM and USER are forwarded. Anything the
server needs must therefore come from its own working directory, not from the
environment it was launched with.
"""

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
