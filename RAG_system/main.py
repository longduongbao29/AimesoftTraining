from fastapi import FastAPI, File, UploadFile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from configs.params import ModelParams
app = FastAPI()
# Khởi tạo các components
model_config = ModelParams()

embeddings = OpenAIEmbeddings(openai_api_key=model_config.openai_api_key)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
vectorstore = Chroma("langchain_store", embeddings) #chroma như database lưu trữ các vector sau khi
llm = OpenAI(temperature=model_config.temperature, openai_api_key=model_config.openai_api_key)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Xử lý và lưu file
    contents = await file.read()
    with open(file.filename, "wb") as f:
        f.write(contents)
    
    # Xử lý tài liệu
    loader = PyPDFLoader(file.filename)
    pages = loader.load_and_split()
    texts = text_splitter.split_documents(pages) # chunking
    
    # Lập chỉ mục
    vectorstore.add_documents(texts)    #dùng chroma để lưu các vector
    
    return {"message": "File uploaded and processed successfully"}

@app.post("/query")
async def query(question: str):
    # Thực hiện truy vấn
    result = qa_chain({"query": question})
    return {"answer": result["result"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
