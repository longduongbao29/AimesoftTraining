from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter
import fitz
from logs.loging import logger
from typing import List
import re

MAX_CHUNK_SIZE = 1000


class TextReader(TextSplitter):
    doc_name: str = None
    file_path: str = None
    text: str = None

    def __init__(self, file_path, doc_name, **kwargs):
        super().__init__(**kwargs)
        self.doc_name = doc_name
        self.file_path = file_path
        self.text = str

    def split_text(self, text: str) -> List[str]:
        """Split text into multiple components."""
        return self.split_text_by_paragraphs(text)

    def readpdf(self):
        doc = fitz.open(self.file_path)
        self.text = ""
        for page in doc:
            self.text += page.get_text()
        doc.close()
        # logger.info({"text:", self.text})
        return self.text

    def create_document(self):
        """Split and create documents from text"""
        doc = Document(page_content=self.text, metadata={"doc_name": self.doc_name})
        docs = self.split_documents([doc])
        return docs

    def split_text_by_paragraphs(self, text):
        """Split text by paragraphs by .\n or \n\n

        Args:
            text (str): text to split

        Returns:
            list: chunks
        """
        paragraphs = re.split(r"\.\n|\n\n", text)
        chunks = []
        chunk = ""
        for paragraph in paragraphs:
            if (
                len(chunk) + len(paragraph) > MAX_CHUNK_SIZE
            ):  # Define your max chunk size
                chunks.append(chunk)
                chunk = paragraph
            else:
                chunk += "\n\n" + paragraph
        if chunk:
            chunks.append(chunk)
        return chunks
