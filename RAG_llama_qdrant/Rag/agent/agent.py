from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate
from Rag.answer.answer import Generate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Your task is make a decision to use retriever_tool or search_tool to answer the given question.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


class Agent:
    def __init__(self, llm, retriever) -> None:
        self.search_tool = TavilySearchResults()
        self.llm = llm
        self.retriever = retriever
        self.retriever_tool = Generate(llm, retriever)
        self.tools = [self.search_tool, self.retriever_tool]
        # self.prompt = hub.pull("hwchase17/openai-functions-agent")
        self.agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent, tools=self.tools, verbose=True
        )

    def run(self, input):
        return self.agent_executor.invoke(input)
