from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import ensure_dir, write_jsonl


QUESTION_KEYS = ("question", "query", "qa_question", "Question", "Query")
ANSWER_KEYS = ("gold_answer", "answer", "target", "Answer", "label")
CONTEXT_KEYS = ("context", "text", "episode", "evidence", "timeline", "passage", "source")


def first_present(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def normalize_episodes(raw_episodes: Any, fallback_context: str, question_id: str) -> list[dict[str, str]]:
    if isinstance(raw_episodes, list) and raw_episodes:
        episodes: list[dict[str, str]] = []
        for index, episode in enumerate(raw_episodes):
            if isinstance(episode, dict):
                episode_id = episode.get("episode_id") or episode.get("id") or f"{question_id}_e{index + 1}"
                text = (
                    episode.get("text")
                    or episode.get("episode_text")
                    or episode.get("content")
                    or episode.get("description")
                    or " ".join(str(value) for value in episode.values() if value is not None)
                )
            else:
                episode_id = f"{question_id}_e{index + 1}"
                text = str(episode)
            episodes.append({"episode_id": str(episode_id), "text": str(text)})
        return episodes

    return [{"episode_id": f"{question_id}_e1", "text": fallback_context}]


def make_record(
    question: str,
    answer: str,
    episodes: list[dict[str, str]],
    evidence_ids: list[str],
    task: str,
    index: int,
) -> dict[str, Any]:
    prefix = "atomic" if task == "atomic" else "multihop"
    question_id = f"{prefix}_{index + 1:06d}"
    return {
        "question_id": question_id,
        "task": task,
        "question": question,
        "gold_answer": answer,
        "episodes": episodes,
        "evidence_episode_ids": evidence_ids,
    }


def iter_json_objects(value: Any) -> Any:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from iter_json_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_json_objects(child)


def sample_from_dict(record: dict[str, Any], task: str, index: int) -> dict[str, Any] | None:
    question = first_present(record, QUESTION_KEYS)
    answer = first_present(record, ANSWER_KEYS)
    if question is None or answer is None:
        return None

    fallback_context = str(first_present(record, CONTEXT_KEYS) or record)
    question_id = f"{task}_{index + 1:06d}"
    episodes = normalize_episodes(record.get("episodes"), fallback_context, question_id)
    evidence_ids = record.get("evidence_episode_ids") or record.get("evidence_ids") or [episodes[0]["episode_id"]]
    evidence_ids = [str(episode_id) for episode_id in evidence_ids]
    return make_record(str(question), str(answer), episodes, evidence_ids, task, index)


def load_json_candidates(path: Path, task: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    samples: list[dict[str, Any]] = []
    for obj in iter_json_objects(data):
        sample = sample_from_dict(obj, task, len(samples))
        if sample:
            samples.append(sample)
    return samples


def load_jsonl_candidates(path: Path, task: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if isinstance(record, dict):
                    sample = sample_from_dict(record, task, len(samples))
                    if sample:
                        samples.append(sample)
    except Exception:
        return []
    return samples


def load_csv_candidates(path: Path, task: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                question = first_present(row, QUESTION_KEYS)
                answer = first_present(row, ANSWER_KEYS)
                if question is None or answer is None:
                    continue
                context = str(first_present(row, CONTEXT_KEYS) or row)
                question_id = f"{task}_{len(samples) + 1:06d}"
                episodes = [{"episode_id": f"{question_id}_e1", "text": context}]
                samples.append(make_record(str(question), str(answer), episodes, [episodes[0]["episode_id"]], task, len(samples)))
    except Exception:
        return []
    return samples


def find_real_samples(task: str) -> list[dict[str, Any]]:
    roots = [PROJECT_ROOT / "data" / "raw", PROJECT_ROOT / "original" / "TimelineQA"]
    candidates: list[dict[str, Any]] = []

    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix == ".jsonl":
                candidates.extend(load_jsonl_candidates(path, task))
            elif suffix == ".json":
                candidates.extend(load_json_candidates(path, task))
            elif suffix == ".csv":
                candidates.extend(load_csv_candidates(path, task))

    return candidates


def make_toy_atomic(n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    foods = ["sushi", "pasta", "salad", "dosa", "sandwich", "noodles"]
    drinks = ["tea", "coffee", "juice", "water"]
    activities = ["read a book", "watched a movie", "went for a walk", "called a friend"]
    records: list[dict[str, Any]] = []

    for index in range(n):
        day = index + 1
        date = f"2010/01/{day:02d}"
        meal = "lunch" if index % 2 == 0 else "dinner"
        other_meal = "dinner" if meal == "lunch" else "lunch"
        food = rng.choice(foods)
        evidence_id = f"e{index + 1}_1"
        episodes = [
            {"episode_id": evidence_id, "text": f"{date}, I ate {food} for {meal}."},
            {"episode_id": f"e{index + 1}_2", "text": f"{date}, I drank {rng.choice(drinks)} during {other_meal}."},
            {"episode_id": f"e{index + 1}_3", "text": f"{date}, I {rng.choice(activities)} in the evening."},
            {"episode_id": f"e{index + 1}_4", "text": f"2010/02/{day:02d}, I discussed travel plans with a friend."},
        ]
        records.append(
            make_record(
                question=f"What did I eat for {meal} on {date}?",
                answer=food,
                episodes=episodes,
                evidence_ids=[evidence_id],
                task="atomic",
                index=index,
            )
        )
    return records


def make_toy_multihop(n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    foods = ["sushi", "pasta", "salad", "dosa"]
    records: list[dict[str, Any]] = []

    for index in range(n):
        target_food = rng.choice(foods)
        count = rng.randint(1, 4)
        episodes: list[dict[str, str]] = []
        evidence_ids: list[str] = []
        for event_index in range(6):
            food = target_food if event_index < count else rng.choice([item for item in foods if item != target_food])
            episode_id = f"m{index + 1}_e{event_index + 1}"
            date = f"2010/03/{event_index + 1:02d}"
            episodes.append({"episode_id": episode_id, "text": f"{date}, I ate {food} during lunch."})
            if food == target_food:
                evidence_ids.append(episode_id)
        records.append(
            make_record(
                question=f"How many times did I eat {target_food} during this week?",
                answer=str(count),
                episodes=episodes,
                evidence_ids=evidence_ids,
                task="multihop",
                index=index,
            )
        )
    return records


def make_toy_samples(task: str, n: int, seed: int) -> list[dict[str, Any]]:
    if task == "atomic":
        return make_toy_atomic(n, seed)
    return make_toy_multihop(n, seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create TimelineQA-style JSONL samples.")
    parser.add_argument("--task", choices=["atomic", "multihop"], required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.n <= 0:
        raise ValueError("--n must be positive.")

    rng = random.Random(args.seed)
    output_path = Path(args.output) if args.output else PROJECT_ROOT / "data" / "samples" / f"{args.task}_n{args.n}.jsonl"

    real_samples = find_real_samples(args.task)
    toy_used = False
    if len(real_samples) >= args.n:
        samples = rng.sample(real_samples, args.n)
        print(f"Loaded {args.n} samples from available TimelineQA-style files.")
    else:
        samples = list(real_samples)
        needed = args.n - len(samples)
        samples.extend(make_toy_samples(args.task, needed, args.seed + len(samples)))
        toy_used = True
        if real_samples:
            print(f"Found only {len(real_samples)} real samples. Added {needed} toy samples to reach n={args.n}.")
        print("WARNING: Real TimelineQA data not found. Created toy samples for pipeline testing only.")

    ensure_dir(output_path.parent)
    write_jsonl(output_path, samples)
    print(f"Wrote {len(samples)} {args.task} samples to {output_path}")


if __name__ == "__main__":
    main()
