import re


def clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


