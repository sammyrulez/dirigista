"""Entrypoint: run the server over stdio as a subprocess of an MCP client."""

from . import telemetry
from .server import mcp


def main() -> None:
    telemetry.configure()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
