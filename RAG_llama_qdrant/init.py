from langchain_community.chat_models.llamacpp import ChatLlamaCpp
from qdrant.client import Qdrant_Client
from langchain_groq import ChatGroq
from Rag.config.config import Config
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

config = Config()


class InitVariable:

    def __init__(self, model_path="models/llama-2-7b-chat.Q2_K.gguf"):
        self.model_path = model_path
        # self.embedding = LlamaCppEmbeddings_(model_path=model_path, n_gpu_layers=15000)
        self.embedding = FastEmbedEmbeddings()

        self.qdrant_client = Qdrant_Client(embeddings=self.embedding)
        # self.llm = ChatLlamaCpp(model_path=model_path, n_gpu_layers=15000, n_ctx=2048)
        self.llm = ChatGroq(api_key=config.groq_api_key)


vars = InitVariable()

print("Initialized variables!")
