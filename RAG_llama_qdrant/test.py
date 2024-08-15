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
from logs.loging import logger

# multi_query = MultiQuery(vars.llm)
# generate = Generate(vars.llm, multi_query)
# print(generate.generate("What is pokemon?"))

# rag_fusion = RAGFusion(vars.llm)
# generate = Generate(vars.llm, retriever= rag_fusion)
# print(generate.generate("What is pokemon?"))


# decomposition = QueryDecompostion(vars.llm, mode="individual")
# generate = Generate(vars.llm, decomposition)
# print(generate.generate("What is pokemon?"))

# stepback = StepBack(model = vars.llm)
# generate = Generate(vars.llm, stepback)
# print(generate.generate("What is pokemon?"))


# hyde = HyDE(vars.llm)
# generate = Generate(vars.llm, hyde)
# print(generate.generate("What is pokemon?"))

# multi_query = MultiQuery(vars.llm)
# agent = Agent(llm=vars.llm, retriever=multi_query)
# print(agent.run({"input": "What is Pokemon?"}))
from Rag.extract_documents.text_reader import TextReader
reader = TextReader("data/test_rag.pdf")

reader.readpdf()
chunks = reader.split_text_by_paragraphs()
logger.info({"chunks":chunks})