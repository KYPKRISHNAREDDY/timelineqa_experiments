from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.denotation_accuracy import denotation_accuracy
from src.evaluation.exact_match import exact_match
from src.evaluation.recall_at_k import recall_at_k
from src.evaluation.token_f1 import token_f1
from src.utils.io import ensure_dir, read_jsonl


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TimelineQA prediction JSONL.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--task", choices=["atomic", "multihop"], required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    prediction_path = Path(args.predictions)
    rows = read_jsonl(prediction_path)
    if not rows:
        raise ValueError("Prediction file is empty.")

    output_path = Path(args.output) if args.output else PROJECT_ROOT / "outputs" / "metrics" / f"{prediction_path.stem}_metrics.json"

    metrics = {
        "task": args.task,
        "model_id": rows[0].get("model_id", ""),
        "backend": rows[0].get("backend", ""),
        "retriever": rows[0].get("retriever", ""),
        "top_k": rows[0].get("top_k", ""),
        "n_examples": len(rows),
        "exact_match": average([exact_match(row.get("predicted_answer"), row.get("gold_answer")) for row in rows]),
        "token_f1": average([token_f1(row.get("predicted_answer"), row.get("gold_answer")) for row in rows]),
        "denotation_accuracy": average(
            [denotation_accuracy(row.get("predicted_answer"), row.get("gold_answer")) for row in rows]
        ),
        "recall_at_1": average(
            [recall_at_k(row.get("retrieved_episode_ids", []), row.get("evidence_episode_ids", []), 1) for row in rows]
        ),
        "recall_at_3": average(
            [recall_at_k(row.get("retrieved_episode_ids", []), row.get("evidence_episode_ids", []), 3) for row in rows]
        ),
        "recall_at_5": average(
            [recall_at_k(row.get("retrieved_episode_ids", []), row.get("evidence_episode_ids", []), 5) for row in rows]
        ),
        "avg_latency_sec": average([float(row.get("latency_sec") or 0.0) for row in rows]),
        "predictions": str(prediction_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
