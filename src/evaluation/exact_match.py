from __future__ import annotations

from src.utils.normalize_answer import normalize_answer


def exact_match(prediction: object, gold_answer: object) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold_answer))
