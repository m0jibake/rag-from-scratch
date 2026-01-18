from openai import AzureOpenAI
import json
from typing import List
from src.mcp.mcp_client import MCPClient
from src.tools import Tool

class Agent:
    def __init__(self, llm_client: AzureOpenAI, mcp_client: MCPClient, tools: List[Tool]):
        self.llm_client = llm_client
        self.mcp_client = mcp_client
        self.tools = tools

    async def get_all_tools(self) -> List[Tool]:
        mcp_tools = await self.mcp_client.list_tools()
        return [tool.to_openai_schema() for tool in self.tools] + mcp_tools
    
    async def execute_tool(self, tool_name: str, args: dict):
        for tool in self.tools:
            if tool.name == tool_name:
                return tool.execute(**args)
        
        return await self.mcp_client.call_tool(tool_name, args)

    async def process_query(self, query: str) -> str:
        messages = [
            {"role": "system", "content": "You have access to tools. For ANY questions about personal information, names, favorites, or user-specific data, you MUST use the search_documents tool. Do NOT answer from memory for such questions."},
            {"role": "user", "content": query}
        ]
        
        available_tools = await self.get_all_tools()
        print(f"🔧 Available tools: {[t['function']['name'] for t in available_tools]}")
        
        response = self.llm_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=available_tools,
        )
        
        message = response.choices[0].message
        print(f"🤖 LLM wants to call: {[tc.function.name for tc in message.tool_calls] if message.tool_calls else 'No tools'}")
        
        if message.tool_calls:
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

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                # Execute the tool (Agent routes to right place)
                result = await self.execute_tool(tool_name, tool_args)
                
                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })
            final_response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
            )
            return final_response.choices[0].message.content or ""

        else: 
            return message.content or ""
        

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nAgent Started!")
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
    import asyncio
    import os
    from dotenv import load_dotenv
    from src.rag_pipeline import RagPipeline
    from src.splitter import SimpleCharacterSplitter
    from src.embeddings import OpenAiEmbeddings
    from src.vector_stores import SimpleVectorStore
    from src.tools import RagTool
    
    load_dotenv()
    
    # Setup LLM client
    llm_client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    
    # Setup RAG
    embedder = OpenAiEmbeddings(llm_client)
    splitter = SimpleCharacterSplitter(chunk_size=500, overlap=100)
    vector_store = SimpleVectorStore(1536)
    rag_pipeline = RagPipeline(client=llm_client, splitter=splitter, embedder=embedder, vector_store=vector_store)
    rag_pipeline.add_documents(["./documents/name.txt", "./documents/favorite_book.pdf"])
    
    # Setup MCP client
    mcp_client = MCPClient()
    await mcp_client.connect_to_server("src/mcp/mcp_server.py")
    
    # Create agent with tools
    rag_tool = RagTool(rag_pipeline)
    agent = Agent(llm_client=llm_client, mcp_client=mcp_client, tools=[rag_tool])
    
    try:
        await agent.chat_loop()
    finally:
        await mcp_client.cleanup()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

