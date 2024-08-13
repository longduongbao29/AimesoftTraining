
from fastapi.responses import HTMLResponse
from langchain.chains.question_answering import load_qa_chain
from schemas.schemas import Question, File
from init import vars, fast_app


@fast_app.get("/", response_class=HTMLResponse)
def read_root():
    return """
        <h2>Hello! Welcome to the model serving api.</h2>
        Check the <a href="/docs">api specs</a>.
    """


@fast_app.post("/predict")
async def model_predict(question: Question):
    chain = load_qa_chain(vars.llm, chain_type="stuff", prompt=vars.qa_prompt)
    try:
        docs = vars.qdrant_client.vectorstore.similarity_search(query=question)
    except Exception as e:
        print(e)
        return e
    answer = chain.invoke(
        {"input_documents": docs, "question": question.question, "context": docs},
        return_only_outputs=True,
    )["output_text"]
    return {"docs": docs, "question": question.question, "answer": answer}


@fast_app.post("/upload_file")
async def upload_to_database(file: File):
    contents = await file.file.read()
    text = contents.decode("utf-8")
    vars.qdrant_client.upload_from_text(text=text, title=file.filename)
    return {"message": "File uploaded and processed successfully"}
