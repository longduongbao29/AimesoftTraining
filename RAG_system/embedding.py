from langchain_openai import AzureOpenAIEmbeddings
from configs.params import ModelParams

model_config = ModelParams()


model_embeddings = AzureOpenAIEmbeddings(
    api_key=model_config.embedding_key,
    azure_endpoint=model_config.embedding_endpoint,
    api_version="2023-07-01-preview",
    azure_deployment="prj-taiho-embedding",
)
