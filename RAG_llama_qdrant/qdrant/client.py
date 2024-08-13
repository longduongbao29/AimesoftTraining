import qdrant_client
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams
from qdrant.text_reader import TextReader
from langchain_text_splitters import CharacterTextSplitter
from Rag.config.config import Config

config = Config()


class Qdrant_Client:
    def __init__(self, embeddings) -> None:

        self.url = config.qdrant_url
        self.api_key = config.qdrant_key
        self.embeddings = embeddings
        self.client = qdrant_client.QdrantClient(
            url=self.url,
            api_key=self.api_key,
        )
        self.text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        try:
            self.client.create_collection(
                collection_name="docs",
                vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
            )
            self.vectorstore = QdrantVectorStore(
                client=self.client,
                collection_name="docs",
                embedding=embeddings,
            )
        except Exception as e:
            print("Error: %s try init from existing collection")
            self.vectorstore = QdrantVectorStore.from_existing_collection(
                url=self.url,
                api_key=self.api_key,
                embedding=embeddings,
                collection_name="docs",
            )

    def upload_docs_to_database(self, docs):
        self.vectorstore.add_documents(docs)

    def retriever(self, text, k=3):
        if len(text) == 0:
            return []
        docs = self.vectorstore.similarity_search(query=text, k=k)
        return docs

    def upload_from_text(self, text, title):
        docs = TextReader.create_document(text, title)
        self.upload_docs_to_database(docs)

    def retriever_map(self, queries):
        docs = []
        for query in queries:
            response = self.retriever(query)
            docs.append(response)
        return docs
