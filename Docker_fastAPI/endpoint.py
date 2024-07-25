from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Question(BaseModel):
    question: str

@app.post("/answer")
def get_answer(question: Question): 
    responses = {
        "What is your name?": "My name is Long",
        "How old are you?": "I am 21 years old"
    }
    answer = responses.get(question.question, "I dont know what u saying")
    return {"answer": answer}
