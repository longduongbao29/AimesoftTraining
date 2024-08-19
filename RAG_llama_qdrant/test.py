# from Rag.answer.answer import Generate
# from Rag.retriever.query_translation import (
#     MultiQuery,
#     RAGFusion,
#     QueryDecompostion,
#     StepBack,
#     HyDE,
# )
# from init import vars
# from Rag.agent.agent import Agent
# from logs.loging import logger

# multi_query = MultiQuery(vars.llm)
# generate = Generate(vars.llm, multi_query)
# print(generate._run("What is pokemon?"))

# rag_fusion = RAGFusion(vars.llm)
# generate = Generate(vars.llm, retriever= rag_fusion)
# print(generate._run("What is pokemon?"))


# decomposition = QueryDecompostion(vars.llm, mode="recursive")
# generate = Generate(vars.llm, decomposition)
# print(generate._run("What is pokemon?"))

# stepback = StepBack(model = vars.llm)
# generate = Generate(vars.llm, stepback)
# print(generate._run("What is pokemon?"))


# hyde = HyDE(vars.llm)
# generate = Generate(vars.llm, hyde)
# print(generate._run("What is pokemon?"))

# multi_query = MultiQuery(vars.llm)
# agent = Agent(llm=vars.llm, retriever=multi_query)
# print(agent.run({"input": "What is the weather tommorow in hanoi?"}))
from init import vars
from langchain_core.documents import Document
from logs.loging import logger


def get_documents():
    client = vars.qdrant_client.client
    collections = client.get_collections().collections
    docs = []
    for collection in collections:
        collection_name = collection.name
        page_size = 100
        offset = 0
        while True:
            response = client.scroll(
                collection_name=collection_name,
                limit=page_size,
                offset=offset,
            )
            for r in response[0]:
                data = r.payload
                doc = Document(
                    metadata=data["metadata"], page_content=data["page_content"]
                )
                docs.append(doc)
            # Nếu số lượng documents trả về ít hơn page_size thì dừng lại
            if len(response[0]) < page_size:
                break

            # Tăng offset cho lần lặp tiếp theo
            offset += page_size
    return docs


logger.output(get_documents())
