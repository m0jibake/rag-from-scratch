import numpy as np
import uuid
import os
from openai import AzureOpenAI
from src.embeddings import OpenAiEmbeddings
from src.vector_stores import SimpleVectorStore, Vector, SimilarityScore
from src.splitter import SimpleCharacterSplitter, Splitter
from dotenv import load_dotenv








class Retriever:
    def __init__(self, vector_store: SimpleVectorStore, embedder: OpenAiEmbeddings) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
    
    def retrieve(self, query: str, k: int, ) -> list[SimilarityScore]:
        query_embedding = self.embedder.embed(query)
        return self.vector_store.search(query_embedding, k)
    
class ContextFormatter:
    def __init__(self):
        pass

    def organize_chunks(self, chunks: list[SimilarityScore]) -> str:
        organized_chunks = "Here are text chunks sorted in descending order by relevancy to the user prompt:"
        
        for i, chunk in enumerate(chunks):
            organized_chunks += f"Chunk {i}: {chunk.vector.text} \n"
        return organized_chunks
    
class Llm:
    def __init__(self, api_key: str, azure_endpoint: str, api_version: str) -> None:
        self.client = AzureOpenAI(
            api_key = api_key,
            azure_endpoint= azure_endpoint,
            api_version = api_version,
            )
        
    def create_response(self, prompt: str, model: str):
        return self.client.chat.completions.create(
            model=model,
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


class RagPipeline:
    # api_key: str, base_url: str, api_version: str,
    def __init__(self, embedder: OpenAiEmbeddings, vector_store: SimpleVectorStore, retriever: Retriever, llm: Llm, splitter: Splitter,  k: int):

        self.vector_store = vector_store
        self.embedder = embedder
        self.retriever = retriever
        self.llm = llm
        self.splitter = splitter
        self.k = k

    def add_documents(self, documents: list[str]):

        for doc in documents:

            chunks = self.splitter.split(doc)
            for chunk in chunks:
                vector = Vector(vector_id=str(uuid.uuid1()), vector=np.array(self.embedder.embed(chunk)), document_id=str(uuid.uuid1()), text=chunk)
                self.vector_store.add(vector)

    def query_rag(self, user_query: str, model: str):
    
        chunks = self.retriever.retrieve(user_query, self.k)
        context = ContextFormatter().organize_chunks(chunks)
        prompt = PromptBuilder(context, user_query).build()
        response = self.llm.create_response(prompt, model)
        return response.choices[0].message.content
    

if __name__ == "__main__":

    _  = load_dotenv()
    API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
    BASE_URL = os.getenv("AZURE_OPENAI_ENDPOINT")

    user_query = "What is your name? Also what is the biggest mountain on Schwäbische Alb?"

    text1 = "Your name is Nate the Great"
    text2 = "I don't like Marshmallows."

    import yaml
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)


    embedder = OpenAiEmbeddings(API_KEY, BASE_URL, API_VERSION)
    vector_store = SimpleVectorStore(num_vector_dimensions=1536)
    retriever = Retriever(vector_store, embedder)
    llm = Llm(API_KEY, BASE_URL, API_VERSION)
    splitter = SimpleCharacterSplitter(500, 2)

    rag = RagPipeline(embedder, vector_store, retriever, llm, splitter, config["retriever"]["k"])
    rag.add_documents([text1, text2])
    
    print(rag.query_rag("What is your name", config["llm"]["model"]))
