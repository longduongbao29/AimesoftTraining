from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)


class TextReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.text = str

    def readfile(self):
        with open(self.file_path, "r") as file:
            self.text = file.read()
        doc = Document(page_content=self.text)
        return doc

    @staticmethod
    def create_document(text, title):
        doc = Document(page_content=text, metadata={"title": title})
        docs = text_splitter.split_documents([doc])
        return docs
