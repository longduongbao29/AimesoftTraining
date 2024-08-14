import Rag.retriever.templates as templates
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from init import vars


class Retriever:
    def __init__(self, model):
        self.generate_prompt = templates.default_prompt
        self.model = model

    def retriever(self, question: str):
        """
        Retrieve documents related to the question using the Qdrant client.
        Args:
        question (str): The question to retrieve related documents.
        Returns:
        List[Document]: A list of documents related to the question.

        """
        docs = vars.qdrant_client.vectorstore.similarity_search(query=question)
        return docs

    def remove_duplicates(self, documents):
        """
        Remove duplicate documents from the list.
        Args:
        documents (List[Document]): The list of documents to remove duplicates from.
        Returns:
        List[Document]: A list of unique documents.
        """
        seen = set()
        unique_documents = []

        for doc in documents:
            id = doc.metadata["_id"]  # Hoặc doc.metadata nếu cần
            if id not in seen:
                seen.add(id)
                unique_documents.append(doc)

        return unique_documents

    def get_page_contents(self, docs):
        """
        Get the page contents of the documents.
        Args:
        docs (List[Document]): The list of documents.
        Returns:
        List[str]: The page contents of the documents.
        """
        page_contents = [doc.page_content for doc in docs]
        return page_contents

    def get_context(self, page_contents):
        """Merge page contents

        Args:
            page_contents (list[str]): page contents from documents.

        Returns:
            str: context
        """
        context = "\n".join(page_contents)
        return context

    def get_input_vars(self, question: str):
        """
        Generate input vars for the prompt.
        Args:
        question (str): The question to retrieve related documents.
        Returns:
        dict: input variables for the prompt.
        """
        docs = self.retriever(question)
        page_contents = self.get_page_contents(docs)
        context = self.get_context(page_contents)
        return {"question": question, "context": context}

    def flatten_docs(self, docs):
        """Flatten documents' array from retrieved documents

        Args:
            docs (list[list[Document]]): retrieved documents

        Returns:
            list[Document]: flattened documents'array
        """
        flatten = []
        for ds in docs:
            for doc in ds:
                flatten.append(doc)
        return flatten
    def generate_queries(self, question: str) -> list[str]:
        """
        Generate 3 queries for the given question
        """
        chain = self.prompt | self.model | StrOutputParser() | (lambda x: x.split("\n"))
        queries = chain.invoke(question)
        return queries

class MultiQuery(Retriever):
    def __init__(self, model) -> None:
        self.model = model
        self.template = templates.multiquery_template
        self.prompt = templates.multiquery_prompt
        self.generate_prompt = templates.default_prompt

  

    def retriever(self, question: str):
        """
        Retrieve documents related to the question using the Qdrant client.
        Args:
        question (str): The question to retrieve related documents.
        Returns:
        List[Document]: A list of documents related to the question.
        """
        queries = self.generate_queries(question)
        docs = vars.qdrant_client.retriever_map(queries)
        docs = self.flatten_docs(docs)
        docs = self.remove_duplicates(docs)
        return docs


class RAGFusion(Retriever):
    def __init__(self, model) -> None:
        self.model = model
        self.template = templates.rag_fusion_template
        self.prompt = templates.rag_fusion_prompt
        self.generate_prompt = templates.default_prompt

    def reciprocal_rank_fusion(self, results, k=60):
        """Reciprocal_rank_fusion that takes multiple lists of ranked documents
        and an optional parameter k used in the RRF formula"""

        # Initialize a dictionary to hold fused scores for each unique document
        fused_scores = {}

        # Iterate through each list of ranked documents
        for docs in results:
            # Iterate through each document in the list, with its rank (position in the list)
            for rank, doc in enumerate(docs):
                # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
                doc_str = dumps(doc)
                # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                # Retrieve the current score of the document, if any
                previous_score = fused_scores[doc_str]
                # Update the score of the document using the RRF formula: 1 / (rank + k)
                fused_scores[doc_str] += 1 / (rank + k)

        # Sort the documents based on their fused scores in descending order to get the final reranked results
        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(
                fused_scores.items(), key=lambda x: x[1], reverse=True
            )
        ]

        # Return the reranked results as a list of tuples, each containing the document and its fused score
        return reranked_results

   

    def retriever(self, question: str):
        """
        Retrieve documents related to the question using the Qdrant client with rerank documents.
        Args:
        question (str): The question to retrieve related documents.
        Returns:
        List[Document]: A list of rerank documents related to the question.
        """
        queries = self.generate_queries(question)
        docs = vars.qdrant_client.retriever_map(queries)
        rerank_docs = self.reciprocal_rank_fusion(docs)
        return rerank_docs

    def get_input_vars(self, question: str):
        """
        Generate input vars for the prompt.
        Args:
        question (str): The question to retrieve related documents.
        Returns:
        dict: input variables for the prompt.
        """
        docs = self.retriever(question)
        docs = [doc[0] for doc in docs]
        page_contents = self.get_page_contents(docs)
        context = self.get_context(page_contents)
        return {"question": question, "context": context}


class QueryDecompostion(Retriever):
    def __init__(self, model, mode) -> None:
        self.model = model
        self.decomposition_mode = mode
        self.template = templates.decomposition_template
        self.prompt = templates.decomposition_prompt
        if mode == "recursive":
            self.generate_prompt = templates.recursive_decomposition_prompt
        else:
            self.generate_prompt = templates.individual_decomposition_prompt



    def format_qa_pair(question, answer):
        """Format Q and A pair"""

        formatted_string = ""
        formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
        return formatted_string.strip()

    def retrieve_and_rag(self, question, prompt_rag, sub_question_generator_chain):
        """RAG on each sub-question"""

        # Use our decomposition /
        sub_questions = sub_question_generator_chain(question)

        # Initialize a list to hold RAG chain results
        rag_results = []

        for sub_question in sub_questions:

            # Retrieve documents for each sub-question
            retrieved_docs = self.retriever(sub_question)

            # Use retrieved documents and sub-question in RAG chain
            answer = (prompt_rag | self.model | StrOutputParser()).invoke(
                {"context": retrieved_docs, "question": sub_question}
            )
            rag_results.append(answer)

        return rag_results, sub_questions

    # Wrap the retrieval and RAG process in a RunnableLambda for integration into a chain


class StepBack(Retriever):
    def __init__(self, model) -> None:
        self.model = model
        self.prompt = templates.step_back_prompt
        self.generate_prompt = templates.generate_step_back_prompt

    def get_input_vars(self, question: str):
        normal_context = self.retriever(question)
        queries = self.generate_queries(question)
        step_back_docs = vars.qdrant_client.retriever_map(queries)
        step_back_docs = self.flatten_docs(step_back_docs)
        docs_content = self.get_page_contents(step_back_docs)
        step_back_context = self.get_context(docs_content)
        input_vars = {
            "normal_context": normal_context,
            "step_back_context": step_back_context,
            "question": question
        }
        return input_vars

class HyDE(Retriever):
    def __init__(self, model) -> None:
        self.model = model
        self.prompt = templates.prompt_hyde
        self.generate_prompt = templates.default_prompt
    def generate_docs(self, question: str) -> list[str]:
        chain = self.prompt | self.model | StrOutputParser() 
        docs = chain.invoke(question)
        return docs
    def retriever(self, question):
        docs_for_retrieval = self.generate_docs(question)
        docs = super().retriever(docs_for_retrieval)
        return docs
