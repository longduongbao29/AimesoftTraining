from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import io

app = FastAPI()

@app.get("/data/")
async def getdata():
	file_path = "/app/data/file.txt"
	with open(file_path, "r", encoding="utf-8") as file:
        	return file.read()
