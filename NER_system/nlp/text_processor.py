# nlp/text_processor.py
from .ner_model import NERModel
from .regex_patterns import RegexPatterns


class TextProcessor:
    def __init__(self):
        self.ner_model = NERModel(model_path="./app/models")
        self.regex_patterns = RegexPatterns()

    def process_text(self, text):
        # Nhận diện thực thể bằng mô hình NER
        entities = self.ner_model.detect_entities(text)

        # Thay thế thực thể nhận diện được bởi mô hình
        processed_text = self.replace_entities(text, entities)

        # Tìm và thay thế các mẫu số postal và thời gian
        postal_patterns = [self.regex_patterns.postal_regex]
        time_patterns = self.regex_patterns.time_regexes

        processed_text = self.regex_patterns.replace_patterns(
            processed_text, postal_patterns, "postal"
        )
        processed_text = self.regex_patterns.replace_patterns(
            processed_text, time_patterns, "date"
        )

        return processed_text

    def replace_entities(self, text, entities):
        sorted_entities = sorted(entities, key=lambda x: x["start"], reverse=True)
        for entity in sorted_entities:
            entity_text = entity["word"]
            entity_label = entity["entity"]
            start = entity["start"]
            end = entity["end"]
            # Thay thế theo vị trí trong văn bản
            text = text[:start] + f"x ({entity_label})" + text[end:]
        return text
