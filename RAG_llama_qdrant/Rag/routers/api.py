
from fastapi.responses import HTMLResponse
from schemas.schemas import File, RetrieverSchema
from Rag.answer.answer import Generate
import Rag.retriever.query_translation as Retrieval
from init import vars, fast_app
from fastapi import APIRouter
router = APIRouter()


def get_retriever(schema: RetrieverSchema):
    if schema.mode


@router.get("/", response_class=HTMLResponse)
def read_root():
    return """
        <h2>Hello! Welcome to the model serving api.</h2>
        Check the <a href="/docs">api specs</a>.
    """

@router.post("/retriever")
def retriever(retrieval: RetrieverSchema):
    question = retrieval.question
    mode = retrieval.mode
    if mode=="defaut"

@router.post("/predict")
async def model_predict(retrieval_schema: RetrieverSchema):
    retriever = get_retriever(retrieval_schema)
    generate = Generate(vars.llm, )
    return 


@router.post("/upload_file")
async def upload_to_database(file: File):
    contents = await file.file.read()
    text = contents.decode("utf-8")
    vars.qdrant_client.upload_from_text(text=text, title=file.filename)
    return {"message": "File uploaded and processed successfully"}

fast_app.include_router(router)