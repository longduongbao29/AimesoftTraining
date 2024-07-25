from fastapi import FastAPI
import requests
import os


app = FastAPI()

ENDPOINT_CONTAINER_PORT = os.getenv('ENDPOINT_CONTAINER_PORT', '7010')
ENDPOINT_URL = f"http://endpoint:{ENDPOINT_CONTAINER_PORT}"

@app.get("/ask")
def ask_question(question: str):
    # chuyển cho endpoint
    response = requests.post(f"{ENDPOINT_URL}/answer", json={"question": question})
    answer = response.json().get("answer")
    return {"question": question, "answer": answer}
