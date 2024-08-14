from fastapi.responses import HTMLResponse
from Rag.schemas.schemas import File, RetrieverSchema, Question
from Rag.answer.answer import Generate, get_retriever

from init import vars
from fastapi import APIRouter

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
    docs = retriever.retriever(question)
    return docs


@router.post("/generate")
async def model_predict(question: Question, retrieval_schema: RetrieverSchema):
    question = question.question
    retriever = get_retriever(retrieval_schema.mode)
    generate = Generate(vars.llm, retriever)
    answer = generate.generate(question)
    return answer


@router.post("/upload_file")
async def upload_to_database(file: File):
    contents = await file.file.read()
    text = contents.decode("utf-8")
    vars.qdrant_client.upload_from_text(text=text, title=file.filename)
    return {"message": "File uploaded and processed successfully"}
