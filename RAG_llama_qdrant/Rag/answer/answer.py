from Rag.retriever.query_translation import (
    Retriever,
    MultiQuery,
    RAGFusion,
    QueryDecompostion,
)
from langchain import hub
from langchain_core.output_parsers import StrOutputParser


class Generate:

    def __init__(self, llm, retriever: Retriever):
        self.retriever = retriever
        self.llm = llm
        self.prompt = retriever.generate_prompt
        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate(self, question: str):
        response = None
        if isinstance(self.retriever, MultiQuery) or isinstance(
            self.retriever, RAGFusion
        ):
            response = self.default_generate(question)
        elif isinstance(self.retriever, QueryDecompostion):
            response = self.decomposition_generate(question)
        return response

    def default_generate(self, question):
        input_vars = self.retriever.get_input_vars(question)
        answer = self.chain.invoke(input_vars)
        return answer

    def decomposition_generate(self, question):
        if self.retriever.decomposition_mode == "recursive":
            questions = self.retriever.generate_queries(question)
            answer = ""
            q_a_pairs = ""
            for q in questions:
                docs = self.retriever.retriever(question)
                page_contents = self.retriever.get_page_contents(docs)
                context = self.retriever.get_context(page_contents)
                answer = self.chain.invoke(
                    {"question": q, "q_a_pairs": q_a_pairs, "context": context}
                )
                q_a_pair = self.retriever.format_qa_pair(q, answer)
                q_a_pairs = q_a_pairs + "\n---\n" + q_a_pair
            return answer
        else:
            prompt_rag = hub.pull("rlm/rag-prompt")
            answers, questions = self.retriever.retrieve_and_rag(
                question, prompt_rag, self.retriever.generate_queries
            )
            context = self.retriever.format_qa_pairs(questions, answers)
            final_answer = self.chain.invoke({"context": context, "question": question})
            return final_answer
