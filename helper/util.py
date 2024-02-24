import logging
import pathlib
import pandas as pd

from typing import Any, List

from langchain.document_loaders import (
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredWordDocumentLoader,
    PDFMinerLoader,
    DirectoryLoader
)
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document



MEMORY = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')

class EpubReader(UnstructuredEPubLoader):
    def __init__(self, file_path: str | List[str], **unstructured_kwargs: Any):
        super().__init__(file_path, **unstructured_kwargs, mode="elements", strategy="fast")

class DocumentLoaderException(Exception):
    def __init__(self, message: str, unsupported_extension: str):
        self.message = message
        self.unsupported_extension = unsupported_extension
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} Unsupported extension: {self.unsupported_extension}"

class DocumentLoader(object):
    supported_extensions = {
        ".pdf": PDFMinerLoader,
        ".txt": TextLoader,
        ".epub": EpubReader,
        ".docx": UnstructuredWordDocumentLoader,
        ".doc": UnstructuredWordDocumentLoader
    }

def load_document(temp_filepath: str) -> List[Document]:
    ext = pathlib.Path(temp_filepath).suffix
    loader_cls = DocumentLoader.supported_extensions.get(ext)

    if not loader_cls:
        raise DocumentLoaderException(
            f"Invalid extension type {ext}, cannot load this type of file",
            unsupported_extension=ext
        )

    loaded = loader_cls(temp_filepath)
    docs = loaded.load()
    logging.info(docs)
    return docs

def load_data(temp_filepath: str) -> List[Document]:
    loaded = loader(temp_filepath, glob="*.pdf", loader_cls=PDFMinerLoader)
    docs = loaded.load()
    logging.info(docs)
    return docs

def load_qna_data(temp_filepath: str):
    qna = pd.read_csv(temp_filepath)
    eval_questions = qna['Questions'].tolist()
    eval_answers = qna['Answers'].tolist()

    examples = [
        {"query": q, "ground_truths": [eval_answers[i]]}
        for i, q in enumerate(eval_questions)
    ]


