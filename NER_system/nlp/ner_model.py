# nlp/ner_model.py
from transformers import pipeline


class NERModel:
    def __init__(self, model_name="tsmatz/xlm-roberta-ner-japanese"):
        self.classifier = pipeline(model=model_name)

    def aggregate_span(self, results, text):
        if results is None:
            return result
        temp = []
        current_result = results[0]

        for result in results[1:]:
            if result["start"] == current_result["end"]:
                current_result["word"] += result["word"]
                current_result["end"] = result["end"]
                current_result["score"] = (
                    current_result["score"] + result["score"]
                ) / 2  # Trung bình điểm số
            else:
                temp.append(current_result)
                current_result = result
        temp.append(current_result)

        sorted_entities = sorted(temp, key=lambda x: x["start"])
        merged_entities = []
        current_entity = None
        for entity in temp:
            if current_entity is None:
                current_entity = entity
            else:
                if entity["start"] <= current_entity["end"]:
                    current_entity["end"] = max(current_entity["end"], entity["end"])
                    current_entity["word"] = current_entity["word"] + entity[
                        "word"
                    ].replace("▁", " ")
                else:
                    merged_entities.append(current_entity)
                    current_entity = entity
        if current_entity is not None:
            merged_entities.append(current_entity)

        ret = []
        current_result = merged_entities[0]
        for result in merged_entities[1:]:
            if (
                current_result["end"] + 1 == result["start"]
                and current_result["entity"] == result["entity"]
            ):
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
        return output

    def detect_entities(self, text):
        return self.ner(text)
