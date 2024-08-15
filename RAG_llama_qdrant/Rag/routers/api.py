from fastapi.responses import HTMLResponse
from Rag.schemas.schemas import File, RetrieverSchema, Question
from Rag.answer.answer import Generate, get_retriever
from Rag.agent.agent import Agent
from init import vars
from fastapi import APIRouter
from fastapi import UploadFile, File
from Rag.extract_documents.text_reader import TextReader

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def read_root():
    return """
        <h2>Hello! Welcome to the model serving api.</h2>
        Check the <a href="/docs">api specs</a>.
    """


@router.post("/retriever")
def retriever(question: Question, mode: RetrieverSchema):
    question = question.question
    retriever = get_retriever(mode.mode)
    docs = retriever.invoke(question)
    return docs


@router.post("/generate")
async def model_predict(question: Question, retrieval_schema: RetrieverSchema):
    question = question.question
    retriever = get_retriever(retrieval_schema.mode)
    # generate = Generate(vars.llm, retriever)
    agent = Agent(vars.llm, retriever)
    answer = agent.run({"input": question})
    return answer


@router.post("/upload_file")
async def upload_to_database(file: UploadFile = File(...)):
    contents = await file.read()
    file_path = "data/{file.filename}"
    text_reader = TextReader(file_path=file_path, doc_name=file.filename)
    with open(file_path, "wb") as f:
        f.write(contents)
    if ".txt" in file.filename:
        text_reader.text = contents.decode("utf-8")
    else:
        text_reader.readpdf()
    vars.qdrant_client.upload_from_text(text_reader)
    return {"message": "File uploaded and processed successfully"}
