from langchain_community.llms import LlamaCpp
from langchain.prompts.prompt import PromptTemplate
from qdrant.client import Qdrant_Client
from Rag.embedding.embedding import LlamaCppEmbeddings_
from fastapi import FastAPI


template = """You are an assistant to the user, you are given some context below, please answer the query of the user with as detail as possible

Context:\"""
{context}
\"""

Question:\"
{question}
\"""

Answer:"""

fast_app = FastAPI()

class InitVariable:
    def __init__(self, model_path="models/llama-2-7b-chat.Q2_K.gguf"):
        self.model_path = model_path
        self.embedding = LlamaCppEmbeddings_(model_path=model_path, n_gpu_layers=15000)
        self.qdrant_client = Qdrant_Client(embeddings=self.embedding)
        self.qa_prompt = PromptTemplate.from_template(template)
        self.llm = LlamaCpp(model_path=model_path, n_gpu_layers=15000, n_ctx=2048)

vars = InitVariable()