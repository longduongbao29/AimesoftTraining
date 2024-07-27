# nlp/ner_model.py
from transformers import pipeline


class NERModel:
    def __init__(self, model_name="tsmatz/xlm-roberta-ner-japanese"):
        self.pipeline = pipeline("ner", model=model_name)

    def detect_entities(self, text):
        return self.pipeline(text)
