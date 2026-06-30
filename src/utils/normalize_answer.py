from __future__ import annotations

import re
import string


_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def normalize_answer(text: object) -> str:
    """Lowercase, remove punctuation, and collapse whitespace."""
    if text is None:
        return ""
    cleaned = str(text).lower().translate(_PUNCT_TABLE)
    return " ".join(cleaned.split())


def extract_number(text: object) -> float | None:
    """Return the first numeric value in text, if one exists."""
    if text is None:
        return None
    match = _NUMBER_RE.search(str(text).replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None
