import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from src.rag_pipeline import RagPipeline

from dotenv import load_dotenv

load_dotenv()

class MCPClient:
    def __init__(self, rag_pipeline: RagPipeline):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.rag_pipeline = rag_pipeline

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


    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [
            {
                "role": "system", 
                "content": "You have access to tools. Always use the search_documents tool if the user query is not about the wheather.",

            },
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        print(".....")
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in response.tools]
        print(available_tools)


        query_type = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }

        available_tools.append({
            "type": "function",
            "function": {
                "name": "search_documents",
                "description": "Search the knowledge base containing user-specific personal information. You MUST use this tool for ANY questions about: names, personal details, preferences, or anything user-specific. This information is NOT in your training data - you must search the documents.",
                "parameters": query_type,
            }
        })


        import os
        from openai import AzureOpenAI
        azure_open_ai = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                
            )
        response = azure_open_ai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": query}],
            tools=available_tools,
        )

        # Process response and handle tool calls
        final_text = []
        message = response.choices[0].message

        # Check if there are tool calls
        if message.tool_calls:
            # Add assistant message to history
            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in message.tool_calls
                ]
            })

            # Execute each tool call
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                import json
                tool_args = json.loads(tool_call.function.arguments)

                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                if tool_name == "search_documents":
                    result_content = self.rag_pipeline.query_rag(tool_args.get('query')).response                
                else:
                # Execute tool call via MCP
                    result = await self.session.call_tool(tool_name, tool_args)
                    result_content = str(result.content)
                
                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_content
                })

            # Get final response from LLM with tool results
            response = azure_open_ai.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                # tools=available_tools
            )

            final_text.append(response.choices[0].message.content)
        else:
            # No tool calls, just return the content
            final_text.append(message.content)

        return "\n".join(final_text)

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

async def main():
    import sys
    import os
    from src.splitter import SimpleCharacterSplitter, Splitter
    from openai import AzureOpenAI
    from src.embeddings import OpenAiEmbeddings
    from src.vector_stores import SimpleVectorStore


    if len(sys.argv) < 2:
        print("Usage: python src/mcpclient.py <server_script_path>")
        print("Example: python src/mcpclient.py src/mcptest.py")
        sys.exit(1)

    _  = load_dotenv()

    azure_client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    embedder = OpenAiEmbeddings(azure_client)
    splitter = SimpleCharacterSplitter(chunk_size=500, overlap=100)
    vector_store = SimpleVectorStore(1536)

    rag_pipeline = RagPipeline(client=azure_client,splitter=splitter,embedder=embedder,vector_store=vector_store)
    rag_pipeline.add_documents(["./documents/name.txt", "./documents/favorite_book.pdf", "./documents/frodo.pdf"])
    
    client = MCPClient(rag_pipeline=rag_pipeline)
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    import sys
    asyncio.run(main())