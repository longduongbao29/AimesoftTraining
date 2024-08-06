from fastapi import FastAPI, File, UploadFile
from configs.params import ModelParams
from chromadb_retrival import ChromaDB
from generate import LLMGenerate

app = FastAPI()
model_config = ModelParams()
chromadb = ChromaDB()
llmgenerate = LLMGenerate()


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    with open(file.filename, "wb") as f:
        f.write(contents)
    chromadb.save2vectorstore(file.filename)
    return {"message": "File uploaded and processed successfully"}


@app.post("/retrieval")
async def query(question: str):
    retriever = chromadb.vectorstore.as_retriever(search_type="mmr")
    found = retriever.invoke(question)
    ret = ""
    for r in found:
        ret += r.page_content + "\n"
    return {"retriever": ret}


@app.post("/bkhn_gen")
async def generate(question: str):
    source_information = await query(question=question)
    inf = source_information["retriever"]
    combined_information = f"From the information below, answer question: \nInformation:\n{inf}.\nQuestion:{question}."
    answer = llmgenerate.generate(combined_information)
    return {"answer": answer}


if __name__ == "__main__":
    import uvicorn

    bkhn_docs_path = "/home/aime/AimesoftTraining/RAG_system/files/QCDT-2023-upload.pdf"
    chromadb.save2vectorstore(path=bkhn_docs_path)

    uvicorn.run(app, host="0.0.0.0", port=1111)
