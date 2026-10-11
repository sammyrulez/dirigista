"""Opt-in tracing of tool calls, off unless OTEL_EXPORTER_OTLP_ENDPOINT is set.

Spans carry metadata only: which tool ran, how long it took, how large the input
was and how confident the answer came back. Never the text being classified, the
caller's categories or criteria, or the chosen category — those are the caller's
content, and a trace backend is one more place they would be stored.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.metadata import version

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, StatusCode
from opentelemetry.util.types import AttributeValue

ENDPOINT_ENV = "OTEL_EXPORTER_OTLP_ENDPOINT"

# Until a provider is installed this is a no-op tracer, so the tools can open
# spans unconditionally and pay nothing when tracing is off.
tracer = trace.get_tracer("dirigista")


def build_provider() -> TracerProvider | None:
    """A provider exporting over OTLP/HTTP, or None when no endpoint is set.

    Only the process environment can switch this on. A .env file is never
    loaded (see `client.resolve_api_key`), so a planted line there can neither
    enable tracing nor point it at another host. The exporter reads the rest of
    the standard OTEL_* variables, headers included, from the same environment.
    """
    if not os.environ.get(ENDPOINT_ENV):
        return None

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )

    provider = TracerProvider(
        resource=Resource.create(
            {"service.name": "dirigista", "service.version": version("dirigista")}
        )
    )
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    return provider


def configure() -> None:
    """Install the provider globally if tracing is enabled; otherwise do nothing."""
    provider = build_provider()
    if provider is not None:
        trace.set_tracer_provider(provider)


@contextmanager
def tool_span(tool: str, attributes: dict[str, AttributeValue]) -> Iterator[Span]:
    """Trace one tool call. `attributes` must be metadata, never caller text.

    The MCP SDK already opens a `tools/call <tool>` span around every call, so
    this one nests inside it and adds what only the tool knows: input sizes and
    how confident the answer was.
    """
    with tracer.start_as_current_span(
        f"classify {tool}",
        attributes={"gen_ai.tool.name": tool, **attributes},
        # The SDK defaults would copy the exception message into the span, and a
        # message can quote the input back. Record the type alone.
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        try:
            yield span
        except Exception as exc:
            span.set_attribute("error.type", type(exc).__qualname__)
            span.set_status(StatusCode.ERROR)
            raise
