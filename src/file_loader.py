from abc import ABC
from uuid import uuid1
from datetime import datetime
from pydantic import BaseModel, Field
from pypdf import PdfReader


class DocumentMetadata(BaseModel):
    file_name: str
    file_path: str
    file_type: str
    extraction_timestamp: datetime
    character_count: int
    page_number: int | None = None

class Document(BaseModel):
    document_id: str = Field(default_factory = lambda : str(uuid1()))
    text: str 
    metadata: DocumentMetadata

class FileLoader(ABC):
    def load(self) -> list[Document]:
        pass


class FileFactory:

    def __init__(self) -> None:
        pass

    def get_loader(self, file_path: str) -> FileLoader:
        file_type = file_path.split(".")[-1]
        if file_type == "txt":
            return TextFileLoader(file_path)
        elif file_type == "pdf":
            return PdfFileLoader(file_path)
        else: 
            raise NotImplementedError


class TextFileLoader(FileLoader):
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def load(self) -> list[Document]:
        with open(self.file_path) as f:
            contents = f.read()

        document = Document(
            text=contents,
            metadata=DocumentMetadata(
                file_name=self.file_path.split("/")[-1],
                file_path=self.file_path,
                file_type=self.file_path.split(".")[-1],
                extraction_timestamp=datetime.now(),
                character_count=len(contents),
                ),
        )

        return [document]
    


class PdfFileLoader(FileLoader):
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def load(self) -> list[Document]:
        reader = PdfReader(self.file_path)
        documents = []
        for i, page in enumerate(reader.pages):
            contents = page.extract_text(0)
            document = Document(
                text=contents,
                metadata=DocumentMetadata(
                    file_name=self.file_path.split("/")[-1],
                    file_path=self.file_path,
                    file_type=self.file_path.split(".")[-1],
                    extraction_timestamp=datetime.now(),
                    character_count=len(contents),
                    page_number=i+1
                    ),

            )
            documents.append(document)
        return documents
