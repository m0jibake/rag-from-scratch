from pydantic import BaseModel, Field, ConfigDict
import uuid
import numpy as np
from numpy.linalg import norm
import operator
from src.file_loader import DocumentMetadata




class Vector(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    vector_id: str = Field(default_factory=lambda: str(uuid.uuid1()))
    vector: np.ndarray
    document_id: str
    text: str
    metadata: DocumentMetadata

class Document(BaseModel):
    text: str
    document_id: str = Field(default_factory=lambda: str(uuid.uuid1()))

class Candidate(BaseModel):
    vector: Vector
    relevance_score: float
    metadata: dict | None = None
    
class SimpleVectorStore:

    def __init__(self, num_vector_dimensions) -> None:
        self.num_vector_dimensions = num_vector_dimensions
        self.vectors: list[Vector] = []

    def add(self, vector: Vector) -> None:
        if len(vector.vector) != self.num_vector_dimensions:
            raise ValueError(f"Error: Provided vector doesn't have the right length. The vector store is configured for vectors of length {self.num_vector_dimensions} but the provided vector has length {len(vector.vector)}")

        self.vectors.append(vector)
        
    def delete(self, vector_id: str):
        self.vectors = [vec for vec in self.vectors if vec.vector_id != vector_id]

    def retrieve(self, vector_id: str) -> Vector | None:
        for vec in self.vectors:
            if vec.vector_id == vector_id:
                return vec
        return None
    
    def count(self)-> int:
        return len(self.vectors)
    
    def search(self, query_vector: np.ndarray, k: int) -> list[Candidate]:
        if len(query_vector) != self.num_vector_dimensions:
            raise ValueError(f"Query vector must be of length {self.num_vector_dimensions}, but is of length {len(query_vector)}")

        q_norm = norm(query_vector) + 1e-10

        cosine_store = []
        for vector in self.vectors:
            v_norm = norm(vector.vector) + 1e-10
            cosine = float(np.dot(query_vector, vector.vector) / (norm(q_norm) * norm(v_norm)) )
            cosine_store.append((vector, cosine))
        cosine_store.sort(key=lambda x: x[1], reverse=True)

        top_k_vectors = cosine_store[:k]
        return [Candidate(vector=vec[0], similarity_score=vec[1]) for vec in top_k_vectors]
        



if __name__ == "__main__":
    vector1 = Vector(vector_id="111", vector=np.array([1,2,3]), document_id="123", text="hi")
    vector2 = Vector(vector_id="222", vector=np.array([100,200,300]), document_id="456", text="animal")
    vs = SimpleVectorStore(num_vector_dimensions=3)


    vs.add(vector1)
    vs.add(vector2)
    print(vs.search(np.array([1,2,4]), 1))




