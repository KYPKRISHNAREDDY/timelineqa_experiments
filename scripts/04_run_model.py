from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.bm25_retriever import BM25Retriever
from src.utils.io import ensure_dir, read_jsonl, read_yaml


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def default_output_path(task: str, sample_path: Path, model_id: str, backend: str, retriever: str) -> Path:
    return PROJECT_ROOT / "outputs" / "predictions" / f"{sample_path.stem}_{safe_name(model_id)}_{backend}_{retriever}.jsonl"


def load_prompt(task: str) -> str:
    prompts = read_yaml(PROJECT_ROOT / "configs" / "prompts.yaml")
    return prompts.get(task) or prompts["atomic"]


def build_context(retrieved: list[dict[str, Any]]) -> str:
    lines = []
    for episode in retrieved:
        lines.append(f"[{episode['episode_id']}] {episode['text']}")
    return "\n".join(lines)


def create_runner(
    backend: str,
    model_id: str,
    max_new_tokens: int,
    temperature: float,
    prompt_template: str,
) -> Any:
    if backend == "hf":
        from src.runners.hf_runner import HFRunner

        return HFRunner(
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            prompt_template=prompt_template,
        )

    if backend == "ollama":
        from src.runners.ollama_runner import OllamaRunner

        return OllamaRunner(
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            prompt_template=prompt_template,
        )

    raise ValueError(f"Unsupported backend: {backend}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one TimelineQA baseline model.")
    parser.add_argument("--task", choices=["atomic", "multihop"], required=True)
    parser.add_argument("--sample", required=True, help="Path to sample JSONL file.")
    parser.add_argument("--model_id", required=True, help="One model id to run.")
    parser.add_argument("--backend", choices=["hf", "ollama"], default="hf")
    parser.add_argument("--retriever", choices=["bm25"], default="bm25")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test only the first N questions.")
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / ".hf_cache"))

    sample_path = Path(args.sample)
    output_path = Path(args.output) if args.output else default_output_path(
        args.task,
        sample_path,
        args.model_id,
        args.backend,
        args.retriever,
    )

    records = read_jsonl(sample_path)
    if args.limit is not None:
        records = records[: args.limit]

    if not records:
        raise ValueError("No samples were found to run.")

    prompt_template = load_prompt(args.task)
    print(f"Loading one model: {args.model_id}")
    runner = create_runner(
        backend=args.backend,
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        prompt_template=prompt_template,
    )

    ensure_dir(output_path.parent)
    created_at = datetime.now(timezone.utc).isoformat()

    with output_path.open("w", encoding="utf-8") as handle:
        for record in tqdm(records, desc="Running model"):
            episodes = record.get("episodes") or []
            if args.retriever == "bm25":
                retriever = BM25Retriever(episodes)
                retrieved = retriever.retrieve(record["question"], top_k=args.top_k)
            else:
                raise ValueError(f"Unsupported retriever: {args.retriever}")

            context = build_context(retrieved)
            start = time.perf_counter()
            predicted_answer = runner.run_model(record["question"], context)
            latency_sec = time.perf_counter() - start

            prediction = {
                "question_id": record.get("question_id"),
                "task": record.get("task", args.task),
                "model_id": args.model_id,
                "backend": args.backend,
                "retriever": args.retriever,
                "top_k": args.top_k,
                "question": record.get("question"),
                "gold_answer": record.get("gold_answer"),
                "predicted_answer": predicted_answer,
                "retrieved_context": context,
                "retrieved_episode_ids": [episode["episode_id"] for episode in retrieved],
                "evidence_episode_ids": record.get("evidence_episode_ids", []),
                "latency_sec": latency_sec,
                "created_at": created_at,
            }
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            handle.flush()

    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
