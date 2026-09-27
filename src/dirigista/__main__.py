"""Entrypoint: run the server over stdio as a subprocess of an MCP client."""

from .server import mcp


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
