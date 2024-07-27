# nlp/regex_patterns.py
import re


class RegexPatterns:
    postal_regex = r"\b\d{3}\s*-\s*\d{4}\b"
    time_regexes = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b",
        r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
        r"\b(?:昭和|平成|令和)\d{1,2}年?\s*\d{1,2}月?\s*\d{1,2}日?\b",
        r"\b(?:昭|平|令)\d{1,2}\s*(?:\.\d{1,2})?\s*(?:\.\d{1,2})?\b",
    ]

    @classmethod
    def find_patterns(cls, text, patterns):
        results = []
        for pattern in patterns:
            results.extend(re.findall(pattern, text))
        return results

    @classmethod
    def replace_patterns(cls, text, patterns, label):
        for pattern in patterns:
            text = re.sub(pattern, lambda m: f"x ({label})", text)
        return text
