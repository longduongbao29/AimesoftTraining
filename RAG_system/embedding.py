from langchain_core.embeddings import Embeddings
from typing import List
from configs.params import ModelParams
import openai

model_config = ModelParams()

class APIEmbedding(Embeddings):
    def __init__(self):
        self.model_embeddings = openai.AzureOpenAI(
            api_key=model_config.embedding_key,
            azure_endpoint=model_config.embedding_endpoint,
            api_version="2023-07-01-preview",
            azure_deployment="prj-taiho-embedding",
        )


    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings.
        """
        embedings = [self.get_embedding_from_api(t).embedding for t in texts]
        return embedings

    def embed_query(self, text: str) -> List[float]:
        """Embed query text.

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        """
        return self.get_embedding_from_api(text).embedding

    def get_embedding_from_api(self, text):
        response = self.model_embeddings.embeddings.create(
            input=text, model="text-embedding-3-large"
        )
        return response.data[0]
