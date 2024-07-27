from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

model_name = "tsmatz/xlm-roberta-ner-japanese"
classifier = pipeline("token-classification", model=model_name)


app = FastAPI()


class Text(BaseModel):
    text: str


@app.post("/predict")
def get_answer(text: Text):
    return {"predictions": classifier(text)}
