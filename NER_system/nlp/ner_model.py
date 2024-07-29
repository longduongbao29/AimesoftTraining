# nlp/ner_model.py
from transformers import pipeline


class NERModel:
    def __init__(self, model_name="tsmatz/xlm-roberta-ner-japanese"):
        self.classifier = pipeline(model=model_name)
    def aggregate_span(self,results, text):
        new_results = []
        current_result = results[0]

        for result in results[1:]:
            if result["start"] == current_result["end"]:
                current_result["word"] += result["word"]
                current_result["end"] = result["end"]
                current_result["score"] = (
                    current_result["score"] + result["score"]
                ) / 2  # Trung bình điểm số
            else:
                new_results.append(current_result)
                current_result = result
        new_results.append(current_result)

        ret = []
        current_result = new_results[0]
        for result in new_results[1:]:
            if text[current_result["end"]] == " " and text[result["end"]] == ",":
                current_result["word"] += result["word"]
                current_result["end"] = result["end"]
                current_result["score"] = (
                    current_result["score"] + result["score"]
                ) / 2  # Trung bình điểm số
            else:
                ret.append(current_result)
                current_result = result
        ret.append(current_result)
        return ret


    def ner(self, text):
        output = self.classifier(text)
        output = self.aggregate_span(output, text)
        return {"text": text, "entities": output}

    def detect_entities(self, text):
        return self.ner(text)
