import os
import numpy as np
from typing import List
from openai import AzureOpenAI
from dotenv import load_dotenv
_ = load_dotenv()

API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
BASE_URL = os.getenv("AZURE_OPENAI_ENDPOINT")
MODEL = "text-embedding-ada-002"
text = "hi there"





class OpenAiEmbeddings:
    def __init__(self, client: AzureOpenAI) -> None:
        self.client = client

    def embed(self, text: str) -> List[np.ndarray]:
        return self.client.embeddings.create(input = [text], model=MODEL ).data[0].embedding
    
    def embed_batch(self, texts: list[str]):
        embeddings = []
        for text in texts:
            embeddings.append(self.embed(text))
        return embeddings
