import qdrant_client
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams
from Rag.extract_documents.text_reader import TextReader
from langchain_text_splitters import CharacterTextSplitter
from Rag.config.config import Config


config = Config()


class Qdrant_Client:
    """Qdrant client for vector databse"""

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
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            self.vectorstore = QdrantVectorStore(
                client=self.client,
                collection_name="docs",
                embedding=embeddings,
            )
        except Exception as e:
            print("Init from existing collection")
            self.vectorstore = QdrantVectorStore.from_existing_collection(
                url=self.url,
                api_key=self.api_key,
                embedding=embeddings,
                collection_name="docs",
            )

    def retriever(self, text: str, k=3):
        """Get k relevant documents to given input text

        Args:
            text (_type_): _description_
            k (int, optional): _description_. Defaults to 3.

        Returns:
            _type_: _description_
        """
        if len(text) == 0:
            return []
        docs = self.vectorstore.similarity_search(query=text, k=k)
        return docs

    def upload_from_text(self, text: TextReader):
        """From input text, chunking and saving to Qdrant collection

        Args:
            text (str): input text
            title (str): title of document
        """
        docs = text.create_documents()
        self.vectorstore.add_documents(docs)

    def retriever_map(self, queries: list[str]) -> list[list]:
        """From input queries, get relevant documents

        Args:
            queries (list[str]): input queries
        Returns:
            list[list]: relevant documents for each query
        """
        docs = []
        for query in queries:
            response = self.retriever(query)
            docs.append(response)
        return docs
        