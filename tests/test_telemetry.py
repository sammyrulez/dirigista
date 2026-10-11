"""Tracing: off by default, switched on only by the environment, metadata only.

The privacy tests matter more than the plumbing ones. A trace backend is one
more place data ends up stored, so the text being classified must never reach a
span — not as an attribute, not in an error.
"""

import threading
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode
from typesafe_sdk import ChoiceAnswer, NoulAnswer, ScoreAnswer

from dirigista import server, telemetry

SECRET = "Mario Rossi owes 4,200 euros"
CATEGORIES = ["unpaid invoice from Acme", "refund request from Acme"]
CRITERIA = ["mildly annoyed at Acme", "furious at Acme"]


@pytest.fixture(scope="module")
def exporter():
    # The global provider can be installed once per process; the module shares it.
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return exporter


@pytest.fixture
def spans(exporter):
    exporter.clear()
    return exporter.get_finished_spans


@pytest.fixture
def backend(fake_client, monkeypatch):
    """Route the tools to a fake client answering all three questions."""

    def install(client):
        @asynccontextmanager
        async def fake_inference_client():
            yield client

        monkeypatch.setattr(server, "inference_client", fake_inference_client)

    install(
        fake_client(
            {
                "verify": NoulAnswer(noul=0.9),
                "categorize": ChoiceAnswer(
                    choice=CATEGORIES[0],
                    confidence=0.8,
                    probabilities={CATEGORIES[0]: 0.8, CATEGORIES[1]: 0.2},
                ),
                "score": ScoreAnswer(
                    score=1.0,
                    confidence=0.7,
                    legend=dict(enumerate(CRITERIA)),
                    probabilities={0: 0.3, 1: 0.7},
                ),
            }
        )
    )
    return install


async def _call_every_tool():
    await server.verify(SECRET, context=SECRET)
    await server.categorize(SECRET, CATEGORIES)
    await server.score(SECRET, CRITERIA)


# --- switching it on --------------------------------------------------------


def test_tracing_is_off_without_an_endpoint(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

    assert telemetry.build_provider() is None


def test_an_endpoint_in_the_environment_turns_tracing_on(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")

    provider = telemetry.build_provider()

    assert provider is not None
    assert provider.resource.attributes["service.name"] == "dirigista"
    provider.shutdown()


def test_an_endpoint_in_a_dotenv_file_does_not(tmp_path, monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    (tmp_path / ".env").write_text("OTEL_EXPORTER_OTLP_ENDPOINT=http://evil:4318\n")
    monkeypatch.chdir(tmp_path)

    assert telemetry.build_provider() is None


# --- authenticating to the collector ---------------------------------------


@pytest.fixture
def collector(tmp_path, monkeypatch):
    """A local OTLP endpoint recording the headers of each export it receives."""
    received: list[dict[str, str]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            received.append({k.lower(): v for k, v in self.headers.items()})
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(
        target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    ).start()
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT", f"http://127.0.0.1:{httpd.server_port}"
    )
    # Neither the developer's shell nor the project's .env may leak in.
    for name in ("OTEL_BEARER_TOKEN", "OTEL_EXPORTER_OTLP_HEADERS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)

    def export_one_span() -> dict[str, str]:
        provider = telemetry.build_provider()
        provider.get_tracer("test").start_span("probe").end()
        provider.shutdown()
        (headers,) = received
        return headers

    yield export_one_span
    httpd.shutdown()


def test_without_a_token_no_authorization_is_sent(collector):
    assert "authorization" not in collector()


def test_a_token_in_the_environment_is_sent_as_a_bearer_header(
    collector, monkeypatch
):
    monkeypatch.setenv("OTEL_BEARER_TOKEN", "token-from-env")

    assert collector()["authorization"] == "Bearer token-from-env"


def test_a_token_in_a_dotenv_file_is_sent_too(collector, tmp_path):
    (tmp_path / ".env").write_text("OTEL_BEARER_TOKEN=token-from-dotenv\n")

    assert collector()["authorization"] == "Bearer token-from-dotenv"


def test_the_token_adds_to_the_standard_headers_instead_of_replacing_them(
    collector, monkeypatch
):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "x-team=search")
    monkeypatch.setenv("OTEL_BEARER_TOKEN", "t")

    headers = collector()

    assert headers["x-team"] == "search"
    assert headers["authorization"] == "Bearer t"


# --- what a span records ----------------------------------------------------


async def test_every_tool_call_produces_one_span_named_after_the_tool(
    backend, spans
):
    await _call_every_tool()

    assert [s.name for s in spans()] == [
        "classify verify",
        "classify categorize",
        "classify score",
    ]


async def test_spans_carry_input_sizes_and_answer_confidence(backend, spans):
    await _call_every_tool()
    verify, categorize, score = (dict(s.attributes) for s in spans())

    assert verify["dirigista.statement.length"] == len(SECRET)
    assert verify["dirigista.context.present"] is True
    assert verify["dirigista.verdict"] == "true"
    assert categorize["dirigista.categories.count"] == 2
    assert categorize["dirigista.confidence"] == 0.8
    assert score["dirigista.criteria.count"] == 2
    assert score["dirigista.level"] == 2


async def test_no_caller_text_ever_reaches_a_span(backend, spans):
    await _call_every_tool()

    recorded = [str(v) for s in spans() for v in s.attributes.values()]
    for text in [SECRET, *CATEGORIES, *CRITERIA]:
        assert not any(text in value for value in recorded), text


async def test_a_failure_records_its_type_but_not_its_message(
    backend, fake_client, spans
):
    class Failing(fake_client):
        async def system_one(self, state, questions, **kwargs):
            raise RuntimeError(f"rejected: {state['statement']}")

    backend(Failing({}))

    with pytest.raises(RuntimeError):
        await server.verify(SECRET)

    (span,) = spans()
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["error.type"] == "RuntimeError"
    assert not span.status.description
    assert not span.events
