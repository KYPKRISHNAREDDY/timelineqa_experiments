from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import ensure_dir, write_jsonl


OFFICIAL_REPO = PROJECT_ROOT / "original" / "TimelineQA"
GENERATED_ROOT = PROJECT_ROOT / "data" / "raw" / "timelineqa_generated"
DEFAULT_CATEGORY = "sparse"
DEFAULT_FINAL_YEAR = 2022


class TimelineQAPrepError(RuntimeError):
    pass


def text_value(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(item) for item in value if item not in (None, ""))
    if value is None:
        return ""
    return str(value)


def discover_official_files() -> dict[str, list[Path]]:
    return {
        "generator": [OFFICIAL_REPO / "src" / "generateDB.py"],
        "atomic_converter": [OFFICIAL_REPO / "src" / "create_qa_data.py"],
        "multihop_generator": [OFFICIAL_REPO / "multihopQA" / "multihopQA.py"],
        "multihop_queryfile": [OFFICIAL_REPO / "multihopQA" / "queryfile.csv"],
        "templates": [OFFICIAL_REPO / "data" / "templates.json"],
        "multihop_seed_csvs": sorted((OFFICIAL_REPO / "data" / "multihop").glob("*.csv"))
        if (OFFICIAL_REPO / "data" / "multihop").exists()
        else [],
    }


def print_discovery() -> None:
    print("Official TimelineQA files discovered:")
    for label, paths in discover_official_files().items():
        existing = [path for path in paths if path.exists()]
        if not existing:
            print(f"  {label}: NOT FOUND")
            continue
        for path in existing:
            print(f"  {label}: {path}")


def require_official_repo() -> None:
    if not OFFICIAL_REPO.exists():
        raise TimelineQAPrepError(
            "Official TimelineQA repo was not found at original/TimelineQA. "
            "Run: bash scripts/00_clone_timelineqa.sh"
        )
    generator = OFFICIAL_REPO / "src" / "generateDB.py"
    if not generator.exists():
        raise TimelineQAPrepError(
            "TimelineQA generator was not found at original/TimelineQA/src/generateDB.py. "
            "Re-clone the official repo with: bash scripts/00_clone_timelineqa.sh"
        )


def generated_dir(seed: int, category: str = DEFAULT_CATEGORY) -> Path:
    return GENERATED_ROOT / f"{category}_seed{seed}"


def generated_json_path(seed: int, category: str = DEFAULT_CATEGORY) -> Path:
    return generated_dir(seed, category) / f"{category}_seed{seed}.json"


def ensure_generated_lifelog(seed: int, category: str = DEFAULT_CATEGORY) -> Path:
    require_official_repo()
    out_dir = generated_dir(seed, category)
    out_json = generated_json_path(seed, category)
    if out_json.exists():
        print(f"Using existing generated TimelineQA lifelog: {out_json}")
        return out_json

    ensure_dir(out_dir)
    generator = OFFICIAL_REPO / "src" / "generateDB.py"
    command = [
        sys.executable,
        str(generator.name),
        "-y",
        str(DEFAULT_FINAL_YEAR),
        "-s",
        str(seed),
        "-c",
        category,
        "-d",
        str(out_dir),
        "-o",
        out_json.name,
    ]
    print("Generating TimelineQA lifelog with official generator...")
    print(" ".join(command))
    result = subprocess.run(
        command,
        cwd=generator.parent,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise TimelineQAPrepError(
            "Official TimelineQA generator failed.\n"
            f"Command: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    if not out_json.exists():
        raise TimelineQAPrepError(f"Generator finished but did not create expected file: {out_json}")
    print(f"Generated lifelog: {out_json}")
    return out_json


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_lifelog(lifelog: dict[str, Any]) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for date in sorted(lifelog.keys()):
        date_events = lifelog.get(date) or {}
        if not isinstance(date_events, dict):
            continue
        for event_name, event in date_events.items():
            if not isinstance(event, dict):
                continue
            episode_id = str(event.get("eid") or f"{date}_{event_name}")
            text = text_value(event.get("text_template_based") or event.get("text_model_based"))
            if not text:
                continue
            episodes.append(
                {
                    "episode_id": episode_id,
                    "text": f"{date}: {text}",
                    "date": date,
                    "event_name": str(event_name),
                    "event": event,
                }
            )
    return episodes


def context_episodes(
    all_episodes: list[dict[str, Any]],
    evidence_episode_ids: list[str],
    max_episodes_per_question: int | None,
    rng: random.Random,
) -> list[dict[str, str]]:
    evidence_set = {str(episode_id) for episode_id in evidence_episode_ids}
    evidence = [episode for episode in all_episodes if episode["episode_id"] in evidence_set]
    distractors = [episode for episode in all_episodes if episode["episode_id"] not in evidence_set]
    rng.shuffle(distractors)

    if max_episodes_per_question and max_episodes_per_question > 0:
        remaining = max(max_episodes_per_question - len(evidence), 0)
        selected = evidence + distractors[:remaining]
    else:
        selected = evidence + distractors

    rng.shuffle(selected)
    return [{"episode_id": episode["episode_id"], "text": episode["text"]} for episode in selected]


def build_atomic_candidates(lifelog_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    lifelog = load_json(lifelog_path)
    lifelog_id = lifelog_path.parent.name
    episodes = flatten_lifelog(lifelog)
    by_episode_id = {episode["episode_id"]: episode for episode in episodes}
    candidates: list[dict[str, Any]] = []

    for episode in episodes:
        event = episode["event"]
        qa_pairs = event.get("atomic_qa_pairs") or []
        for qa_index, qa_pair in enumerate(qa_pairs):
            if not isinstance(qa_pair, list | tuple) or len(qa_pair) < 2:
                continue
            question = text_value(qa_pair[0]).strip()
            answer = text_value(qa_pair[1]).strip()
            if not question or not answer:
                continue
            evidence_id = episode["episode_id"]
            if evidence_id not in by_episode_id:
                continue
            candidates.append(
                {
                    "question": question,
                    "gold_answer": answer,
                    "evidence_episode_ids": [evidence_id],
                    "data_source": "real_timelineqa",
                    "lifelog_id": lifelog_id,
                    "original_question_id": f"{lifelog_id}:{evidence_id}:atomic:{qa_index}",
                    "question_type": "atomic",
                }
            )

    return candidates, episodes, lifelog_id


def parse_qa_db(path: Path, task: str) -> list[dict[str, Any]]:
    data = load_json(path)
    text_rows = data.get("text") or []
    episodes = []
    for index, row in enumerate(text_rows):
        if isinstance(row, list | tuple) and len(row) >= 4:
            date, event_name, episode_id, text = row[:4]
        else:
            continue
        episodes.append(
            {
                "episode_id": str(episode_id),
                "text": f"{date}: {text_value(text)}",
                "date": str(date),
                "event_name": str(event_name),
            }
        )

    key = "atomic-questions" if task == "atomic" else "multihop-questions"
    records = []
    for index, question_row in enumerate(data.get(key) or []):
        evidence_list = question_row.get("evidence_list") or []
        evidence_ids = [str(item[2]) for item in evidence_list if isinstance(item, list | tuple) and len(item) >= 3]
        answer = question_row.get("answer")
        records.append(
            {
                "question": text_value(question_row.get("question")).strip(),
                "gold_answer": format_answer(answer),
                "evidence_episode_ids": evidence_ids,
                "data_source": "real_timelineqa",
                "lifelog_id": text_value(data.get("name") or path.stem),
                "original_question_id": f"{path.stem}:{task}:{index}",
                "question_type": task,
                "_episodes": episodes,
            }
        )
    return [record for record in records if record["question"] and record["gold_answer"] and record["evidence_episode_ids"]]


def format_answer(answer: Any) -> str:
    if isinstance(answer, list):
        if len(answer) == 1 and isinstance(answer[0], list) and len(answer[0]) == 1:
            return text_value(answer[0][0]).strip()
        flattened = []
        for item in answer:
            if isinstance(item, list):
                flattened.append(" ".join(text_value(part) for part in item))
            else:
                flattened.append(text_value(item))
        return "; ".join(part.strip() for part in flattened if part.strip())
    return text_value(answer).strip()


def find_existing_qa_db_records(task: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    search_roots = [PROJECT_ROOT / "data" / "raw", PROJECT_ROOT / "original" / "TimelineQA"]
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("db-*.json"):
            try:
                records.extend(parse_qa_db(path, task))
            except Exception as exc:
                print(f"Skipping unreadable QA DB file {path}: {exc}")
    return records


def ensure_multihop_artifacts(seed: int, category: str = DEFAULT_CATEGORY) -> Path:
    lifelog_path = ensure_generated_lifelog(seed, category)
    directory = lifelog_path.parent
    queries_csv = directory / "queries.csv"
    if queries_csv.exists():
        return directory

    multihop_script = OFFICIAL_REPO / "multihopQA" / "multihopQA.py"
    queryfile = OFFICIAL_REPO / "multihopQA" / "queryfile.csv"
    if not multihop_script.exists() or not queryfile.exists():
        raise TimelineQAPrepError(
            "Official multihop generator files are missing. Expected "
            "original/TimelineQA/multihopQA/multihopQA.py and queryfile.csv."
        )

    command = [sys.executable, str(multihop_script.name), "-q", str(queryfile.name), "-d", str(directory)]
    print("Generating TimelineQA multihop query artifacts with official script...")
    result = subprocess.run(
        command,
        cwd=multihop_script.parent,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise TimelineQAPrepError(
            "Could not generate multihop artifacts. The official script requires optional packages "
            "`pandasql` and `numpyencoder`. Install them or create queries.csv/q*-result.csv manually.\n"
            f"Command: {' '.join(command)}\n"
            f"stdout:\n{result.stdout[-4000:]}\n"
            f"stderr:\n{result.stderr[-4000:]}"
        )
    if not queries_csv.exists():
        raise TimelineQAPrepError(f"Multihop script finished but did not create {queries_csv}")
    return directory


def split_date_values(date_values: list[Any]) -> tuple[list[str], list[str], list[int]]:
    years: list[str] = []
    months: list[str] = []
    days: list[int] = []
    for value in date_values:
        parts = str(value).split(" ")[0].split("/")
        if len(parts) == 1:
            parts = str(value).split(" ")[0].split("-")
        if len(parts) >= 3:
            years.append(str(int(parts[0])))
            months.append(str(int(parts[1])))
            days.append(int(parts[2]))
        else:
            years.append("")
            months.append("")
            days.append(0)
    return years, months, days


def evidence_for_multihop_row(directory: Path, lifelog: dict[str, Any], row: dict[str, str]) -> list[dict[str, Any]]:
    import pandas as pd

    params = json.loads(row.get("params") or "{}")
    datafile = (row.get("datafiles") or "").split(",")[0].strip()
    if not datafile:
        return []

    csv_path = directory / datafile
    if not csv_path.exists():
        return []

    df_file = pd.read_csv(csv_path)
    date_column = "date"
    columns = set(df_file.columns)
    if date_column not in columns:
        if datafile == "travel_dining-log.csv" and "dining_date" in columns:
            date_column = "dining_date"
        elif "place_visit_date" in columns:
            date_column = "place_visit_date"
        elif "start_date" in columns or "start_year" in params:
            date_column = "start_date"
        else:
            return []

    years, months, days = split_date_values(list(df_file[date_column]))
    df_file["year"] = years
    df_file["month"] = months
    df_file["day"] = days

    filtered = df_file
    question = row.get("question") or ""
    for param, param_value_raw in params.items():
        if param not in set(filtered.columns) and param != "start_year":
            continue
        param_value = str(param_value_raw)
        column = "year" if param == "start_year" else param
        if "%" in param_value or "friend" in column or "people" in column:
            filtered = filtered[filtered[column].astype(str).str.contains(param_value, na=False)]
        elif (" since " in question or " after " in question) and column == "year":
            filtered = filtered[filtered[column].astype(str) >= param_value]
        else:
            filtered = filtered[filtered[column].astype(str) == param_value]

    event_ids = {str(event_id) for event_id in filtered.get("eid", [])}
    dates = {str(date).split(" ")[0].replace("-", "/") for date in filtered.get(date_column, [])}
    evidence = []
    for date in dates:
        for event_name, event in (lifelog.get(date) or {}).items():
            if str(event.get("eid")) in event_ids:
                text = text_value(event.get("text_template_based") or event.get("text_model_based"))
                evidence.append(
                    {
                        "episode_id": str(event.get("eid")),
                        "text": f"{date}: {text}",
                        "date": date,
                        "event_name": str(event_name),
                        "event": event,
                    }
                )
    return evidence


def result_answer(directory: Path, q_id: str, answer_column: str) -> str:
    if answer_column == "?":
        return ""
    result_path = directory / f"{q_id}-result.csv"
    if not result_path.exists():
        return ""
    answer_columns = [int(part) for part in answer_column.split(",") if part.strip().isdigit()]
    if not answer_columns:
        return ""
    start = answer_columns[0]
    end = answer_columns[-1]
    answers: list[str] = []
    with result_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) <= start:
                continue
            answers.append(" ".join(cell for cell in row[start : end + 1] if cell != ""))
    return "; ".join(answer for answer in answers if answer)


def build_multihop_candidates(seed: int, category: str = DEFAULT_CATEGORY) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    existing_records = find_existing_qa_db_records("multihop")
    if existing_records:
        record = existing_records[0]
        return existing_records, record.pop("_episodes"), record.get("lifelog_id", "qa_db")

    directory = ensure_multihop_artifacts(seed, category)
    lifelog_path = generated_json_path(seed, category)
    lifelog = load_json(lifelog_path)
    lifelog_id = directory.name
    episodes = flatten_lifelog(lifelog)

    queries_csv = directory / "queries.csv"
    candidates: list[dict[str, Any]] = []
    with queries_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            q_id = row.get("q_id") or f"q{index}"
            answer = result_answer(directory, q_id, row.get("answer_column") or "")
            evidence = evidence_for_multihop_row(directory, lifelog, row)
            evidence_ids = [episode["episode_id"] for episode in evidence]
            question = (row.get("question") or "").strip()
            if not question or not answer or not evidence_ids:
                continue
            candidates.append(
                {
                    "question": question,
                    "gold_answer": answer,
                    "evidence_episode_ids": evidence_ids,
                    "data_source": "real_timelineqa",
                    "lifelog_id": lifelog_id,
                    "original_question_id": f"{lifelog_id}:{q_id}",
                    "question_type": row.get("q_id") or "multihop",
                }
            )
    return candidates, episodes, lifelog_id


def make_toy_fallback(task: str, n: int, seed: int) -> list[dict[str, Any]]:
    import importlib.util

    toy_script = PROJECT_ROOT / "scripts" / "02_make_samples.py"
    spec = importlib.util.spec_from_file_location("toy_samples", toy_script)
    if spec is None or spec.loader is None:
        raise TimelineQAPrepError("Could not import toy sample generator.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = module.make_toy_samples(task, n, seed)
    for record in records:
        record["data_source"] = "toy_fallback"
    return records


def make_records(
    candidates: list[dict[str, Any]],
    all_episodes: list[dict[str, Any]],
    task: str,
    n: int,
    seed: int,
    max_episodes_per_question: int | None,
) -> list[dict[str, Any]]:
    if len(candidates) < n:
        raise TimelineQAPrepError(
            f"Only found {len(candidates)} real {task} questions, but --n requested {n}. "
            "Try a smaller --n or generate more TimelineQA lifelogs."
        )

    rng = random.Random(seed)
    selected = rng.sample(candidates, n)
    records: list[dict[str, Any]] = []
    for index, candidate in enumerate(selected):
        question_id = f"real_{task}_{index + 1:06d}"
        evidence_ids = [str(episode_id) for episode_id in candidate["evidence_episode_ids"]]
        episodes = context_episodes(all_episodes, evidence_ids, max_episodes_per_question, rng)
        records.append(
            {
                "question_id": question_id,
                "task": task,
                "question": candidate["question"],
                "gold_answer": candidate["gold_answer"],
                "episodes": episodes,
                "evidence_episode_ids": evidence_ids,
                "data_source": candidate.get("data_source", "real_timelineqa"),
                "lifelog_id": candidate.get("lifelog_id"),
                "original_question_id": candidate.get("original_question_id"),
                "question_type": candidate.get("question_type"),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare real TimelineQA samples in project JSONL format.")
    parser.add_argument("--task", choices=["atomic", "multihop"], required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max_episodes_per_question", type=int, default=100)
    parser.add_argument("--allow_toy_fallback", action="store_true")
    args = parser.parse_args()

    if args.n <= 0:
        raise ValueError("--n must be positive.")

    print_discovery()
    output_path = Path(args.output) if args.output else PROJECT_ROOT / "data" / "samples" / f"real_{args.task}_n{args.n}.jsonl"

    try:
        if args.task == "atomic":
            lifelog_path = ensure_generated_lifelog(args.seed)
            candidates, episodes, lifelog_id = build_atomic_candidates(lifelog_path)
        else:
            candidates, episodes, lifelog_id = build_multihop_candidates(args.seed)

        print(f"Loaded lifelog_id={lifelog_id}")
        print(f"Found {len(episodes)} real TimelineQA episodes.")
        print(f"Found {len(candidates)} real {args.task} question candidates.")
        records = make_records(
            candidates=candidates,
            all_episodes=episodes,
            task=args.task,
            n=args.n,
            seed=args.seed,
            max_episodes_per_question=args.max_episodes_per_question,
        )
    except Exception as exc:
        if not args.allow_toy_fallback:
            raise TimelineQAPrepError(
                f"Could not prepare real TimelineQA {args.task} data: {exc}\n\n"
                "Real experiments do not use toy fallback automatically.\n"
                "For atomic data, make sure original/TimelineQA exists and run:\n"
                "  bash scripts/00_clone_timelineqa.sh\n"
                "Then rerun this script.\n"
                "For multihop data, generate official multihop query artifacts first or install optional "
                "packages used by original/TimelineQA/multihopQA/multihopQA.py.\n"
                "For pipeline testing only, pass --allow_toy_fallback."
            ) from exc

        print(f"WARNING: Real TimelineQA data preparation failed: {exc}")
        print("WARNING: Using explicit toy fallback because --allow_toy_fallback was provided.")
        records = make_toy_fallback(args.task, args.n, args.seed)

    ensure_dir(output_path.parent)
    write_jsonl(output_path, records)
    print(f"Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
