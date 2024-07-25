from fastapi import FastAPI
import requests

app = FastAPI()

@app.get("/ask")
def ask_question(question: str):
    # chuyển cho service2
    response = requests.post("http://service2:7010/answer", json={"question": question})
    answer = response.json().get("answer")
    return {"question": question, "answer": answer}
