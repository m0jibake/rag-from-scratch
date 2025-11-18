from pydantic import BaseModel, Field
import uuid



class Vector(BaseModel):
    vector_id: str = Field(default_factory=lambda: str(uuid.uuid1()))
    # document_id: str
    vector: list[float]
    # text: str

class Document(BaseModel):
    text: str
    document_id: str = Field(default_factory=lambda: str(uuid.uuid1()))
    
class SimpleVectorStore:

    def __init__(self, num_vector_dimensions) -> None:
        self.num_vector_dimensions = num_vector_dimensions
        self.vectors: list[Vector] = []
        # self.vector_metadata

    def add(self, vector: Vector) -> None:
        if len(vector.vector) > self.num_vector_dimensions:
            raise ValueError(f"Error: Provided vector is too long. The vector store is configured for vectors of length {self.num_vector_dimensions} but the provided vector has length {len(vector)}")
        elif len(vector.vector) < self.num_vector_dimensions:    
            raise ValueError(f"Error: Provided vector is too short. The vector store is configured for vectors of length {self.num_vector_dimensions} but the provided vector has length {len(vector)}")

        self.vectors.append(vector)
        
    def delete(self, vector_id: str):
        self.vectors = [vec for vec in self.vectors if vec.vector_id != vector_id]

    def retrieve(self, vector_id: str) -> list[Vector]:
        return [vec for vec in self.vectors if vec.vector_id == vector_id]
    
    def count(self)-> int:
        return len(self.vectors)
    
    def search():
        raise NotImplementedError   
    


if __name__ == "__main__"
    vector = Vector(vector_id="111", vector=[1,2,3])
    vs = SimpleVectorStore(num_vector_dimensions=3)

    print(vs.count())
    vs.add(vector)
    print(vs.count())
    vs.delete("111")
    print(vs.count())



