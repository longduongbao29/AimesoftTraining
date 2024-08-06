from fastapi import FastAPI, File, UploadFile
import global_vars


app = FastAPI()


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    with open(file.filename, "wb") as f:
        f.write(contents)
    global_vars.chromadb.save2vectorstore(file.filename)
    return {"message": "File uploaded and processed successfully"}


@app.post("/retrieval")
async def query(question: str):
    found = global_vars.chromadb.retriever.invoke(question)
    ret = ""
    for r in found:
        ret += r.page_content + "\n"
    return {"retriever": ret}


@app.post("/bkhn_gen")
async def generate(question: str):
    source_information = await query(question=question)
    inf = source_information["retriever"]
    prompt = f"From the information below, answer question: \nInformation:\n{inf}.\nQuestion:{question}."
    answer = global_vars.llmgenerate.generate(prompt)
    return {"answer": answer}


if __name__ == "__main__":
    import uvicorn

    bkhn_docs_path = "/home/aime/AimesoftTraining/RAG_system/files/QCDT-2023-upload.pdf"
    global_vars.chromadb.save2vectorstore(path=bkhn_docs_path)

    uvicorn.run(app, host="0.0.0.0", port=1111)
