from abc import ABC, abstractmethod

from src.rag_pipeline import RagPipeline

class Tool(ABC):

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def description(self):
        pass

    @property
    @abstractmethod
    def parameters(self):
        pass

    @abstractmethod
    def execute(self, **kwargs):
        pass

    def to_openai_schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }
    

class RagTool(Tool):
    def __init__(self, rag_pipeline: RagPipeline):
        self.rag_pipeline = rag_pipeline

    def execute(self, **kwargs):
        query = kwargs.get("query")
        response = self.rag_pipeline.query_rag(query)
        return response.response
    
    @property
    def name(self):
        return "search_documents"
    
    @property
    def description(self):
        return "Tool to use whenever the query is not about the weather."
    
    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }


    

