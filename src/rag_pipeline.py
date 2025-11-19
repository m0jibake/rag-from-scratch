import numpy as np
import os
from openai import AzureOpenAI
from src.embeddings import OpenAiEmbeddings
from src.vector_stores import SimpleVectorStore, Vector

API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
BASE_URL = os.getenv("AZURE_OPENAI_ENDPOINT")

user_query = "What is your name? Also what is the biggest mountain on Schwäbische Alb?"






class Retriever:
    def __init__(self, vector_store: SimpleVectorStore, embedder: OpenAiEmbeddings) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
    
    def retrieve(self, query: str, k: int, ) -> list[tuple[Vector, float]]:
        query_embedding = self.embedder.embed(query)
        return self.vector_store.search(query_embedding, k)
    
class ContextFormatter:
    def __init__(self):
        pass

    def organize_chunks(self, chunks: list[tuple[Vector, float]]) -> str:
        organized_chunks = "Here are text chunks sorted in descending order by relevancy to the user prompt:"
        
        for i, chunk in enumerate(chunks):
            organized_chunks += f"Chunk {i}: {chunk[0]} \n"
        return organized_chunks
    
class Llm:
    def __init__(self, api_key: str, azure_endpoint: str, api_version: str) -> None:
        self.client = AzureOpenAI(
            api_key = api_key,
            azure_endpoint= azure_endpoint,
            api_version = api_version,
            )
        
    def create_response(self, prompt: str):
        return self.client.chat.completions.create(
            model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]

        )
    
class PromptBuilder:
    def __init__(self, context: str, user_query: str):
        self.system_prompt = """
            You are a helpful assistant. Answer based on the provided context.
            If the context doesn't contain relevant information, say so.
            """
        self.context = context
        self.user_query = user_query

    def build(self):
        return f"""{self.system_prompt}

            Context:
            {self.context}

            User Question:
            {self.user_query}

            Answer:"""

if __name__ == "__main__":
# if True:
    vs = SimpleVectorStore(num_vector_dimensions=1536)
    embedder = OpenAiEmbeddings(API_KEY, BASE_URL, API_VERSION)
    text1 = "Your name is Nate the Great"
    text2 = "I don't like Marshmallows."
    vector1 = Vector(vector_id="111", vector=np.array(embedder.embed(text1)), document_id="123", text=text1)
    vector2 = Vector(vector_id="222", vector=np.array(embedder.embed(text2)), document_id="456", text=text2)



    vs.add(vector1)
    vs.add(vector2)
    retriever = Retriever(vs, embedder)
    chunks = retriever.retrieve(user_query, 1)
    context = ContextFormatter().organize_chunks(chunks)
    prompt = PromptBuilder(context, user_query).build()
    response = Llm(API_KEY, BASE_URL, API_VERSION).create_response(prompt)
    print(response.choices[0].message.content)
