from __future__ import annotations

import re
import unicodedata


def normalize_words(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return " ".join(text.split())
