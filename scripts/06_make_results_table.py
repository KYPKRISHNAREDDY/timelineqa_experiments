from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLUMNS = [
    "task",
    "model_id",
    "backend",
    "retriever",
    "top_k",
    "n_examples",
    "exact_match",
    "token_f1",
    "denotation_accuracy",
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "avg_latency_sec",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine metrics JSON files into one CSV table.")
    parser.add_argument("--metrics_dir", default=str(PROJECT_ROOT / "outputs" / "metrics"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "tables" / "baseline_results.csv"))
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    rows = []
    for path in sorted(metrics_dir.glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"Skipping invalid metrics file: {path}")
            continue
        rows.append({column: row.get(column) for column in COLUMNS})

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(output_path, index=False)
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
