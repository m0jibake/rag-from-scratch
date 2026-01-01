# RAG from Scratch

A production-style Retrieval-Augmented Generation (RAG) system built from first principles, implementing core patterns found in LangChain and LlamaIndex.

## Overview

This project implements a modular RAG pipeline with composable components, supporting both synchronous and asynchronous execution modes, streaming responses, and two-stage retrieval with reranking.

## Features

### Core Capabilities

- **Vector-based retrieval**: Cosine similarity search over embedded documents
- **Two-stage retrieval**: Initial retrieval (k=20) followed by cross-encoder reranking (top-5)
- **Streaming responses**: Token-by-token streaming from LLM with backpressure support
- **Async execution**: Full async/await support for concurrent operations
- **Composable pipeline**: Unix-style pipe operator for chaining components

### Architecture

- **Runnable abstraction**: Base class providing `invoke`, `ainvoke`, `stream`, and `astream` methods
- **State-based flow**: Shared `PipelineState` object passed through pipeline stages
- **Event-driven streaming**: Typed `StreamEvent` objects for observability
- **Document processing**: Support for text and PDF files with configurable chunking

### Components

- `RetrieverRunnable`: Vector similarity search
- `RerankerRunnable`: Cross-encoder scoring for relevance refinement
- `ContextFormatterRunnable`: Context assembly for LLM prompt
- `PromptBuilderRunnable`: Final prompt construction
- `LlmRunnable`: Azure OpenAI integration with streaming support

## Installation

```bash
uv lock && uv sync
```

## Configuration

Create a `.env` file with Azure OpenAI credentials:

```
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
```

## Usage

### Basic Query

```python
from openai import AzureOpenAI
from src.rag_pipeline import RagPipeline
from src.embeddings import OpenAiEmbeddings
from src.splitter import SimpleCharacterSplitter
from src.vector_stores import SimpleVectorStore

client = AzureOpenAI(api_key=..., api_version=..., azure_endpoint=...)
embedder = OpenAiEmbeddings(client)
splitter = SimpleCharacterSplitter(chunk_size=500, overlap=100)
vector_store = SimpleVectorStore(dimensions=1536)

rag = RagPipeline(
    client=client,
    splitter=splitter,
    embedder=embedder,
    vector_store=vector_store
)

# Index documents
rag.add_documents(["./documents/file1.txt", "./documents/file2.pdf"])

# Query
result = rag.query_rag("What is the main topic?")
print(result.response)
```

### Async Streaming

```python
import asyncio

async def stream_query():
    async for event in rag.stream_query("What is Python?"):
        if event.type == "llm_token":
            print(event.data["token"], end="", flush=True)
        elif event.type == "final":
            final_state = event.data["state"]
    return final_state

result = asyncio.run(stream_query())
```

### Custom Pipeline

```python
from src.rag_pipeline import Chain, RetrieverRunnable, RerankerRunnable

custom_pipeline = (
    RetrieverRunnable(vector_store)
    | RerankerRunnable(top_k=3)
    | CustomProcessorRunnable()
    | LlmRunnable(client, model="gpt-4o")
)

state = PipelineState(query="...", query_as_vector=...)
result = custom_pipeline.invoke(state)
```

## Pipeline Stages

1. **Retrieval**: Embedding-based similarity search retrieves top-20 candidates
2. **Reranking**: Cross-encoder (ms-marco-MiniLM-L6-v2) reranks to top-5
3. **Context Formatting**: Assembles chunks with metadata into context string
4. **Prompt Building**: Constructs final LLM prompt with context and query
5. **Generation**: Azure OpenAI generates response with optional streaming

## Project Structure

```
src/
├── rag_pipeline.py       # Main pipeline and runnable components
├── vector_stores.py      # Vector storage and similarity search
├── embeddings.py         # OpenAI embedding integration
├── splitter.py           # Text chunking
├── file_loader.py        # Document loading (text, PDF)
└── simple_rag_pipeline.py # Legacy implementation
```

## Technical Details

### Retrieval

- Embedding model: OpenAI text-embedding-ada-002 (1536 dimensions)
- Similarity metric: Cosine similarity
- Initial retrieval: Top-20 candidates

### Reranking

- Model: cross-encoder/ms-marco-MiniLM-L6-v2
- Purpose: Improves precision by joint encoding of query-document pairs
- Output: Relevance scores for final ranking

### Streaming

- Implements pull-based streaming with natural backpressure
- Consumer controls flow rate through async iteration
- Events propagate through entire pipeline chain

## Design Patterns

- **Runnable Interface**: Unified API for sync/async, streaming/non-streaming execution
- **Pipe Operator**: `|` operator for component composition
- **State Object**: Immutable-friendly state passing between components
- **Event System**: Typed events for observability and debugging
