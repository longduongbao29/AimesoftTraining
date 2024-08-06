from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from configs.params import ModelParams
import os

model_config = ModelParams()
os.environ["TAVILY_API_KEY"] = model_config.tavily_key


class SearchOnline:
    def __init__(self):
        self.tavily = TavilySearchResults()

    def search(self, query):
        response = self.tavily.invoke(query)
        contents = ''
        for r in response:
            contents += r["content"]+"\n"
        return contents



