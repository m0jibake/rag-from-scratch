import numpy as np
from datetime import datetime
import uuid
import os
from openai import AzureOpenAI
from src.embeddings import OpenAiEmbeddings
from src.vector_stores import SimpleVectorStore, Vector, Candidate
from src.splitter import SimpleCharacterSplitter, Splitter
from src.file_loader import TextFileLoader, Document, FileFactory
from dotenv import load_dotenv
from abc import ABC, abstractmethod
from typing import Optional, Iterator, AsyncIterator
from pydantic import BaseModel, Field
import asyncio


class StreamEvent(BaseModel):
    type: str
    data: dict
    timestamp: datetime
    metadata: dict

class PipelineState(BaseModel):
    query: str = Field(..., description="User query")
    candidates: Optional[list[Candidate]] = Field(None, description="Retrieved chunks")
    organized_chunks: Optional[str] = Field(None, description="Formatted context")
    prompt: Optional[str] = Field(None, description="Final prompt")
    response: Optional[str] = Field(None, description="LLM response")
    query_as_vector: Optional[np.ndarray] = Field(None, description="The query vector as embedding")

    # Metadata
    retrieval_time: Optional[datetime] = None
    formatting_time: Optional[datetime] = None
    llm_time: Optional[datetime] = None

    class Config:
        arbitrary_types_allowed = True


class Runnable(ABC):
    """Base class for composable pipeline components."""
    
    @abstractmethod
    def invoke(self, state: PipelineState) -> PipelineState:
        """Execute the component with shared state."""
        pass

    async def ainvoke(self, state: PipelineState) -> PipelineState:
        """Asynchronous non-streaming execution."""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.invoke, state)

    def stream(self, state: PipelineState) -> Iterator[StreamEvent]:
        """Synchronous, streaming execution."""
        result = self.invoke(state)
        yield StreamEvent(
            type="final",
            data={"state": result},
            timestamp=datetime.now(),
            metadata={"component": self.__class__.__name__}
        )

    async def astream(self, state: PipelineState) -> AsyncIterator[StreamEvent]:
        """Asynchronous, streaming execution."""
        result = await self.ainvoke(state)
        yield StreamEvent(
            type="final",
            data={"state": result},
            timestamp=datetime.now(),
            metadata={"component": self.__class__.__name__}
        )

    def __or__(self, other: 'Runnable') -> 'Chain':
        """Support pipe operator: runnable1 | runnable2"""
        return Chain([self, other])

class Chain(Runnable):
    """Orchestrates a series of Runnables."""
    def __init__(self, runnables: list[Runnable]):
        self.runnables = runnables

    def invoke(self, state: PipelineState) -> PipelineState:
        """Execute all runnables in sequence, passing state through."""
        for runnable in self.runnables:
            state = runnable.invoke(state)
        return state
    
    async def ainvoke(self, state: PipelineState) -> PipelineState:
        """Sequential async execution"""
        for runnable in self.runnables:
            state = await runnable.ainvoke(state)
        return state
    
    def stream(self, state: PipelineState) -> Iterator[StreamEvent]:
        """Sequential setreaming, yield all events."""
        for runnable in self.runnables:
            for event in runnable.stream(state):
                yield event

                if event.type == "final":
                    state = event.data["state"]
    
    async def astream(self, state: PipelineState) -> AsyncIterator[StreamEvent]:
        """Sequential async streaming, yiel all events."""
        for runnable in self.runnables:
            async for event in runnable.astream(state):
                yield event

                if event.type == "final":
                    state = event.data["state"]






class RetrieverRunnable(Runnable):
    def __init__(self, vector_store: SimpleVectorStore):
        self.vector_store = vector_store

    def invoke(self, state: PipelineState) -> PipelineState:
        """
        Retrieve documents matching the query.
        
        Reads from state: "query"
        Writes to state: "candidates"
        """
        query = state.query_as_vector
        retrieved_docs = self.vector_store.search(query, k=20)
        state.candidates = retrieved_docs
        return state



    
class ContextFormatterRunnable(Runnable):
    def __init__(self):
        pass

    def invoke(self, state: PipelineState) -> PipelineState:
        chunks: list[Candidate] = state.candidates
        organized_chunks = "Here are text chunks sorted in descending order by relevancy to the user prompt:"
        
        for i, chunk in enumerate(chunks):
            page_info = f"Page {chunk.vector.metadata.page_number}" if chunk.vector.metadata.page_number else "N/A"
            relevancy_pct = f"{chunk.relevance_score * 100:.0f}%"
            organized_chunks += f"Chunk {i} ({chunk.vector.metadata.file_name}, Page: {page_info}, Relevancy: {relevancy_pct}): {chunk.vector.text} \n"
        
        state.organized_chunks = organized_chunks
        return state



    
class LlmRunnable(Runnable):
    def __init__(self, client: AzureOpenAI, model: str) -> None:
        self.client = client
        self.model = model
        
    def invoke(self, state: PipelineState) -> PipelineState:
        """
        Send prompt to LLM and get response.
        
        Reads from state: "prompt"
        Writes to state: "response"
        """
        prompt = state.prompt
        try: 
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            state.response = response.choices[0].message.content
            state.llm_time = datetime.now().isoformat()

        except Exception as e:
            state.response= f"Error querying LLM: {str(e)}"

        return state
    
    async def astream(self, state: PipelineState) -> AsyncIterator[StreamEvent]:
        """Streaming: yield tokens as they arrive."""
        cumulative = ""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": state.prompt}],
            stream=True,
        )
        for chunk in response:
            if chunk.choices and len(chunk.choices) > 0:
                token = chunk.choices[0].delta.content
                if token: 
                    cumulative += token

                    yield StreamEvent(
                        type="llm_token",
                        data={"token": token, "cumulative": cumulative},
                        timestamp=datetime.now(),
                        metadata={"component": "LlmRunnable"},
                    )

        state.response = cumulative
        state.llm_time = datetime.now()
        yield StreamEvent(
            type="final",
            data={"state": state},
            timestamp= datetime.now(),
            metadata={"component": "LlmRunnable"},
        )

    
class PromptBuilderRunnable(Runnable):
    def __init__(self):
        pass

    def invoke(self, state: PipelineState) -> PipelineState:
        """
        Build a prompt combining query and context.
        
        Reads from state: "query", "organized_chunks"
        Writes to state: "prompt"
        """
        query = state.query
        context = state.organized_chunks
        prompt = f"""You are a helpful assistant. Answer the user's question based on the provided context.

            CONTEXT:
            {context}

            USER QUESTION:
            {query}

            ANSWER:"""
                    
        state.prompt = prompt
        return state


class RerankerRunnable(Runnable):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2" , top_k: int = 5):
        from sentence_transformers.cross_encoder import CrossEncoder
        self.model = CrossEncoder(model_name)
        self.top_k = top_k

    def invoke(self, state: PipelineState) -> PipelineState:
        
        cross_encoder_similarity_scores = self.model.predict([[state.query, c.vector.text] for c in state.candidates])

        for candidate, reranking_score in zip(state.candidates, cross_encoder_similarity_scores):

            candidate.relevance_score = float(reranking_score)

        state.candidates.sort(key=lambda x: x.relevance_score, reverse=True)
    
        state.candidates = state.candidates[:self.top_k]
        return state

        




    
class RagPipeline:
    def __init__(self, 
                client: AzureOpenAI,
                splitter: Splitter,
                embedder: OpenAiEmbeddings,
                vector_store: SimpleVectorStore,     
                model: str = "gpt-4o",
                chunk_size: int = 500,
                chunk_overlap: int = 100,
                retrieval_k: int = 5
    ):
        self.client = client
        self.splitter = splitter
        self.embedder = embedder
        self.vector_store = vector_store
        self.model = model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retrieval_k = retrieval_k

        self.file_factory = FileFactory()

        self._pipeline: Chain = (
            RetrieverRunnable(self.vector_store)
            | RerankerRunnable()
            | ContextFormatterRunnable()
            | PromptBuilderRunnable()
            | LlmRunnable(client, model)
        )

    def add_documents(self, files: list[str]):

        for file in files:
            documents = FileFactory().get_loader(file).load()
            for document in documents:
                chunks = self.splitter.split(document.text)
                for chunk in chunks:
                    vector = Vector(vector_id=str(uuid.uuid1()), vector=np.array(self.embedder.embed(chunk)), document_id=document.document_id, text=chunk, metadata=document.metadata)
                    self.vector_store.add(vector)


    def query_rag(self, user_query: str):
        state = PipelineState(query=user_query, query_as_vector=np.array(self.embedder.embed(user_query)))
        final_state = self._pipeline.invoke(state)
        return final_state
    
    async def aquery_rag(self, user_query: str) -> PipelineState:
        """Async execution (non-streaming, full response at end)."""
        state = PipelineState(
            query=user_query,
            query_as_vector=np.array(self.embedder.embed(user_query))
        )
        return await self._pipeline.ainvoke(state)
    
    async def stream_query(self, user_query: str) -> AsyncIterator[StreamEvent]:
        """Async streaming (progressive token yields)."""
        state = PipelineState(
            query=user_query,
            query_as_vector=np.array(self.embedder.embed(user_query))
        )
        async for event in self._pipeline.astream(state):
            yield event


    


if __name__ == "__main__":

    _  = load_dotenv()
    API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
    BASE_URL = os.getenv("AZURE_OPENAI_ENDPOINT")

    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    embedder = OpenAiEmbeddings(client)
    splitter = SimpleCharacterSplitter(chunk_size=500, overlap=100)
    vector_store = SimpleVectorStore(1536)



    # user_query = "What is your name? Also what is the biggest mountain on Schwäbische Alb?"
    # user_query = "give a summary of the boing report" 
    user_query = "what is my favorite animal ?"
    user_query = "What is your name?"




    import yaml
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)


    
    rag = RagPipeline(client=client,splitter=splitter,embedder=embedder,vector_store=vector_store)


    rag.add_documents(["./documents/name.txt", "./documents/favorite_book.pdf", "./documents/frodo.pdf"])
    
    # final_state = rag.query_rag(user_query)
    # print(final_state.response)

    # ✅ Wrap async code in async function
    async def test_streaming():
        print("=" * 60)
        print("🔄 STREAMING TEST")
        print("=" * 60)
        
        async for event in rag.stream_query(user_query):
            if event.type == "llm_token":
                print(event.data["token"], end="", flush=True)
                await asyncio.sleep(0.05)  # 50ms delay per token as otherwise streaming is not too fast for the eyes
            elif event.type == "final":
                print("\n[Complete]")
                final_state = event.data["state"]
        return final_state
    
    # ✅ Run async function from sync context
    import asyncio
    final_state = asyncio.run(test_streaming())
    print(f"\nFull response: {final_state.response}")