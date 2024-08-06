from langchain_text_splitters import CharacterTextSplitter
from embedding import APIEmbedding
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader


class ChromaDB:
    def __init__(self):
        self.text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        self.vectorstore = Chroma
        self.embedding = APIEmbedding()

    def save2vectorstore(self, path):
        loader = PyPDFLoader(file_path=path)
        documents = loader.load_and_split()
        docs = self.text_splitter.split_documents(documents)
        for doc in docs:
            doc.page_content = doc.page_content.replace("\n", " ")
        # Extract text from documents for embedding
        self.vectorstore = Chroma.from_documents(
            documents=docs, embedding=self.embedding
        )
