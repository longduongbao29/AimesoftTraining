from configs.params import ModelParams
from langchain.agents import create_tool_calling_agent, initialize_agent
from langchain import hub
from online_search import SearchOnline
import global_vars
from langchain.agents import AgentExecutor
from langchain_core.utils.function_calling import format_tool_to_openai_function
from langchain_openai import AzureChatOpenAI

# Get the prompt to use - you can modify this!

model_config = ModelParams()


class LLMGenerate:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            api_key=model_config.chat_key,
            azure_endpoint=model_config.chat_endpoint,
            azure_deployment="prj-taiho-gpt4o",
            openai_api_version="2024-02-01",
        )
        self.prompt = hub.pull("hwchase17/openai-functions-agent")
        self.search_online = SearchOnline()
        self.search = self.search_online.tavily
        self.tools = [self.search, global_vars.chromadb.retriever_tool]
        # self.model_with_tools = self.llm.bind_tools(self.tools)
        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent, tools=self.tools, verbose=True
        )

    def generate(self, content):
        response = self.agent_executor.invoke({"input": content})
        # response = self.llm.invoke(content)
        return response["output"]


