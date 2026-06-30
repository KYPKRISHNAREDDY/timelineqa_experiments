from __future__ import annotations

from src.utils.normalize_answer import extract_number, normalize_answer


def denotation_accuracy(
    prediction: object,
    gold_answer: object,
    tolerance: float = 1e-3,
) -> float:
    pred_number = extract_number(prediction)
    gold_number = extract_number(gold_answer)

    if pred_number is not None and gold_number is not None:
        return float(abs(pred_number - gold_number) <= tolerance)

    return float(normalize_answer(prediction) == normalize_answer(gold_answer))
