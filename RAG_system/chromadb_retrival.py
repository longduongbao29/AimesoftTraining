from langchain_text_splitters import CharacterTextSplitter
from embedding import model_embeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.tools.retriever import create_retriever_tool
from langchain_core.vectorstores import VectorStoreRetriever


class ChromaDB:
    def __init__(self):
        self.text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        self.vectorstore = Chroma
        self.embedding = model_embeddings
        self.retriever = VectorStoreRetriever
        self.retriever_tool = create_retriever_tool(
            self.retriever,
            "online_search",
            "Search for information online",
        )

    def save2vectorstore(self, path):
        loader = PyPDFLoader(file_path=path)
        documents = loader.load_and_split()
        docs = self.text_splitter.split_documents(documents)
        # for doc in docs:
        #     doc.page_content = doc.page_content.replace("\n", " ")
        # Extract text from documents for embedding
        self.vectorstore = Chroma.from_documents(
            documents=docs, embedding=self.embedding
        )
        self.retriever = self.vectorstore.as_retriever(search_type="mmr")
