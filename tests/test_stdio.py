"""End-to-end check that the server is usable over stdio by a real MCP client."""

import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_a_client_can_handshake_and_list_tools_over_stdio():
    params = StdioServerParameters(command=sys.executable, args=["-m", "dirigista"])

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()

    assert {tool.name for tool in result.tools} == {"verify", "categorize", "score"}
