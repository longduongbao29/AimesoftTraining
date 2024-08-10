# nlp/regex_patterns.py
import re


class RegexPatterns:
    postal_regex = r"\b\d{3}\s*-\s*\d{4}\b"
    time_regexes = [
        r"\b(3[01]|[12][0-9]|0?[1-9])(\/|-| )(1[0-2]|0?[1-9])(\/|-| )([0-9]{4})\b",
        r"\b([0-9]{4})(\/|-| )(1[0-2]|0?[1-9])(\/|-| )(3[01]|[12][0-9]|0?[1-9])\b",
        r"\b(昭和|平成|令和)([0-9]{1})(年)([0-9]{1})(月)([0-9]{1})(日)\b",
        r"\b(昭|平|令)([0-9]*.)([0-9]*.)([0-9]*)\b",
    ]

    @classmethod
    def find_patterns(cls, text, patterns):
        ret = []
        for pattern in patterns:
            all_matches = re.findall(pattern, text)
            for match in all_matches:
                ret.append("".join(match))
        return ret

    @classmethod
    def replace_patterns(cls, text, patterns, label):
        for pattern in patterns:
            text = re.sub(pattern, lambda m: f"x ({label})", text)
        return text
