import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


from dotenv import load_dotenv

load_dotenv()

class MCPClient:
    def __init__(self,):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self, server_script_path: str) -> None:

        server_params = StdioServerParameters(
            command="python",
            args=[server_script_path],
            env=None
        )


        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])


    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


    async def list_tools(self):
        response = await self.session.list_tools()
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in response.tools]
        return available_tools
    
    async def call_tool(self, tool_name: str, tool_args: dict):
        result = await self.session.call_tool(tool_name, tool_args)
        result_content = str(result.content)
        return result_content

# async def main():
#     import sys


#     if len(sys.argv) < 2:
#         print("Usage: python src/mcpclient.py <server_script_path>")
#         print("Example: python src/mcpclient.py src/mcptest.py")
#         sys.exit(1)

#     _  = load_dotenv()


    
#     client = MCPClient()
#     try:
#         await client.connect_to_server(sys.argv[1])
#         await client.chat_loop()
#     finally:
#         await client.cleanup()


# if __name__ == "__main__":
#     import sys
#     asyncio.run(main())