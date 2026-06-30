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
DF1_SAMPLE_QUERY_IDS = {"q2", "q3", "q4", "q6", "q10", "q16", "q18", "q23", "q24", "q25"}
DF_SAMPLE_QUERY_IDS = {
    "q7",
    "q8",
    "q9",
    "q11",
    "q12",
    "q13",
    "q14",
    "q15",
    "q19",
    "q26",
    "q28",
    "q29",
    "q30",
    "q31",
    "q32",
    "q39",
    "q40",
    "q41",
}
MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


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


def official_multihop_command(directory: Path) -> str:
    return (
        "cd original/TimelineQA/multihopQA && "
        f"python multihopQA.py -q queryfile.csv -d \"{directory}\""
    )


def generated_lifelog_command(seed: int, directory: Path, category: str = DEFAULT_CATEGORY) -> str:
    return (
        "cd original/TimelineQA/src && "
        f"python generateDB.py -y {DEFAULT_FINAL_YEAR} -s {seed} -c {category} "
        f"-d \"{directory}\" -o {category}_seed{seed}.json"
    )


def read_queryfile_rows(queryfile: Path) -> list[dict[str, str]]:
    if not queryfile.exists():
        raise TimelineQAPrepError(
            f"Missing official multihop query template file: {queryfile}\n"
            "It should be provided by the official TimelineQA repo. Run:\n"
            "  bash scripts/00_clone_timelineqa.sh"
        )

    rows: list[dict[str, str]] = []
    with queryfile.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for index, row in enumerate(reader):
            if len(row) < 8:
                print(f"Skipping malformed queryfile row {index}: {row}")
                continue
            rows.append(
                {
                    "q_id": row[0],
                    "query": row[1],
                    "datafiles": row[2],
                    "variables": row[3],
                    "question": row[4],
                    "params": row[5],
                    "answer_column": row[6],
                    "answer_type": row[7],
                }
            )
    return rows


def normalize_date_string(value: Any) -> str:
    text = str(value).split(" ")[0]
    separator = "/" if "/" in text else "-"
    parts = text.split(separator)
    if len(parts) < 3:
        return text
    try:
        return f"{int(parts[0]):04d}/{int(parts[1]):02d}/{int(parts[2]):02d}"
    except ValueError:
        return text.replace("-", "/")


def prepare_sql_dates(df: Any, date_columns: list[str]) -> Any:
    import pandas as pd

    for column in date_columns:
        if column not in df.columns:
            continue
        parsed = pd.to_datetime(df[column], errors="coerce")
        df[column] = parsed.dt.strftime("%Y-%m-%d %H:%M:%S").fillna(df[column].astype(str))
    return df


def add_year_month_day(df: Any, date_column: str, year_name: str = "year", month_name: str = "month") -> Any:
    import pandas as pd

    if date_column not in df.columns:
        return df
    parsed = pd.to_datetime(df[date_column], errors="coerce")
    df[year_name] = parsed.dt.year.astype("Int64")
    df[month_name] = parsed.dt.month.astype("Int64")
    df["day"] = parsed.dt.day.astype("Int64")
    return df


def process_multihop_data_file(datafile: str, directory: Path) -> tuple[Any, Any]:
    import pandas as pd

    csv_path = directory / datafile
    if not csv_path.exists():
        raise TimelineQAPrepError(
            f"Missing generated TimelineQA log file: {csv_path}\n"
            "This file should be created by the official lifelog generator.\n"
            f"Run:\n  {generated_lifelog_command(0, directory).replace('-s 0', '-s <seed>')}"
        )

    df = pd.read_csv(csv_path)
    df_flat = df.copy()

    if datafile == "daily_chat-log.csv" and "friends" in df_flat.columns:
        df_flat["friends"] = list(df_flat["friends"].fillna("").astype(str).str.split(","))
        df_flat = df_flat.explode("friends")
        df_flat["friends"] = df_flat["friends"].astype(str).str.strip()
    elif datafile == "daily_meal-log.csv" and "people_string" in df_flat.columns:
        df_flat["people_string"] = list(df_flat["people_string"].fillna("").astype(str).str.split(","))
        df_flat = df_flat.explode("people_string")
        df_flat["people_string"] = df_flat["people_string"].astype(str).str.strip()
    elif datafile == "weekly_bakeorcook-log.csv" and "cuisine" in df_flat.columns:
        df_flat["cuisine"] = list(df_flat["cuisine"].fillna("").astype(str).str.split(","))
        df_flat = df_flat.explode("cuisine")
        df_flat["cuisine"] = df_flat["cuisine"].astype(str).str.strip()
    elif datafile == "weekly_grocery-log.csv" and "fruits" in df_flat.columns:
        df_flat["fruits"] = list(df_flat["fruits"].fillna("").astype(str).str.split(","))
        df_flat = df_flat.explode("fruits")
        df_flat["fruits"] = df_flat["fruits"].astype(str).str.strip()
    elif datafile == "weekly_hobby-log.csv" and "people_string" in df_flat.columns:
        df_flat["people_string"] = list(df_flat["people_string"].fillna("").astype(str).str.split(","))
        df_flat = df_flat.explode("people_string")
        df_flat["people_string"] = df_flat["people_string"].astype(str).str.strip()

    if datafile == "marriages-log.csv":
        df = add_year_month_day(df, "married_date")
        df_flat = add_year_month_day(df_flat, "married_date")
        df = prepare_sql_dates(df, ["married_date"])
        df_flat = prepare_sql_dates(df_flat, ["married_date"])
    elif datafile == "travel-log.csv":
        df = add_year_month_day(df, "start_date", year_name="start_year", month_name="start_month")
        df_flat = add_year_month_day(df_flat, "start_date", year_name="start_year", month_name="start_month")
        df = prepare_sql_dates(df, ["start_date", "end_date"])
        df_flat = prepare_sql_dates(df_flat, ["start_date", "end_date"])
    elif datafile in {"travel_dining-log.csv", "travel_places_visited-log.csv"}:
        df = add_year_month_day(df, "start_date", year_name="start_year", month_name="start_month")
        df_flat = add_year_month_day(df_flat, "start_date", year_name="start_year", month_name="start_month")
        date_cols = ["start_date", "end_date", "dining_date", "place_visit_date"]
        df = prepare_sql_dates(df, date_cols)
        df_flat = prepare_sql_dates(df_flat, date_cols)
    else:
        df = add_year_month_day(df, "date")
        df_flat = add_year_month_day(df_flat, "date")
        df = prepare_sql_dates(df, ["date"])
        df_flat = prepare_sql_dates(df_flat, ["date"])

    return df, df_flat


def to_python_value(value: Any) -> Any:
    try:
        import pandas as pd

        if pd.isna(value):
            return ""
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def sample_row(df: Any, rng: random.Random) -> Any:
    if len(df) == 0:
        raise TimelineQAPrepError("Cannot sample variables from an empty generated log dataframe.")
    return df.sample(1, random_state=rng.randint(0, 2**31 - 1)).iloc[0]


def query_variables(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def apply_query_variables(row: dict[str, str], df: Any, df_flat: Any, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
    q_id = row["q_id"]
    query = row["query"]
    question = row["question"]
    variables = query_variables(row.get("variables") or "")
    format_dict: dict[str, Any] = {}

    if variables:
        source = df_flat if q_id in DF1_SAMPLE_QUERY_IDS else df if q_id in DF_SAMPLE_QUERY_IDS else None
        if source is not None:
            sampled = sample_row(source, rng)
            for variable in variables:
                if variable not in sampled.index:
                    continue
                value = to_python_value(sampled[variable])
                if variable == "people" and isinstance(value, str) and "," in value:
                    people = [person.strip() for person in value.split(",") if person.strip()]
                    if people:
                        value = rng.choice(people)
                format_dict[variable] = value
            query = query.format(**format_dict)

    for placeholder in sorted(format_dict):
        token = "{" + placeholder + "}"
        value = format_dict[placeholder]
        if placeholder == "month":
            try:
                value = MONTH_NAMES[int(value) - 1]
            except Exception:
                pass
        question = question.replace(token, str(value))

    params = {}
    if row.get("params", "").strip():
        params.update(json.loads(row["params"]))
    params.update(format_dict)
    return query, question, params


def execute_sql_query(query: str, tables: dict[str, Any]) -> Any:
    import pandas as pd
    import sqlite3

    conn = sqlite3.connect(":memory:")
    try:
        for table_name, dataframe in tables.items():
            if dataframe is not None and not isinstance(dataframe, str):
                dataframe.to_sql(table_name, conn, index=False, if_exists="replace")
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()


def generate_multihop_artifacts(directory: Path, seed: int) -> None:
    import pandas as pd

    queryfile = OFFICIAL_REPO / "multihopQA" / "queryfile.csv"
    rows = read_queryfile_rows(queryfile)
    rng = random.Random(seed)
    queries_data = {
        "q_id": [],
        "query": [],
        "params": [],
        "question": [],
        "datafiles": [],
        "answer_column": [],
        "answer_type": [],
    }

    print("Generating TimelineQA multihop query artifacts in project data directory.")
    print(f"Equivalent official command: {official_multihop_command(directory)}")
    for row in rows:
        q_id = row["q_id"]
        datafiles = [item.strip() for item in row["datafiles"].split(",") if item.strip()]
        if not datafiles:
            continue

        df, df_flat = process_multihop_data_file(datafiles[0], directory)
        df2 = df3 = None
        if len(datafiles) > 1:
            df2, df3 = process_multihop_data_file(datafiles[1], directory)

        query, question, params = apply_query_variables(row, df, df_flat, rng)
        result = execute_sql_query(query, {"df": df, "df1": df_flat, "df2": df2, "df3": df3})
        result.to_csv(directory / f"{q_id}-result.csv")

        queries_data["q_id"].append(q_id)
        queries_data["query"].append(query)
        queries_data["params"].append(json.dumps(params, ensure_ascii=False))
        queries_data["question"].append(question)
        queries_data["datafiles"].append(",".join(datafiles))
        queries_data["answer_column"].append(row["answer_column"])
        queries_data["answer_type"].append(row["answer_type"])

    pd.DataFrame(data=queries_data).to_csv(directory / "queries.csv", index=False)


def missing_multihop_result_files(directory: Path) -> list[Path]:
    queries_csv = directory / "queries.csv"
    if not queries_csv.exists():
        return [queries_csv]

    missing: list[Path] = []
    with queries_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if (row.get("answer_column") or "") == "?":
                continue
            result_path = directory / f"{row.get('q_id')}-result.csv"
            if not result_path.exists():
                missing.append(result_path)
    return missing


def ensure_multihop_artifacts(seed: int, category: str = DEFAULT_CATEGORY) -> Path:
    lifelog_path = ensure_generated_lifelog(seed, category)
    directory = lifelog_path.parent

    missing = missing_multihop_result_files(directory)
    if not missing:
        return directory

    multihop_script = OFFICIAL_REPO / "multihopQA" / "multihopQA.py"
    queryfile = OFFICIAL_REPO / "multihopQA" / "queryfile.csv"
    missing_official = [path for path in [multihop_script, queryfile] if not path.exists()]
    if missing_official:
        raise TimelineQAPrepError(
            "Official multihop generator files are missing:\n"
            + "\n".join(f"  {path}" for path in missing_official)
            + "\nRun:\n  bash scripts/00_clone_timelineqa.sh"
        )

    try:
        generate_multihop_artifacts(directory, seed)
    except Exception as exc:
        raise TimelineQAPrepError(
            "Could not generate multihop artifacts from official query templates.\n"
            f"Missing or stale artifact example: {missing[0]}\n"
            f"Expected location: {directory}\n"
            "The official script that normally creates these files is:\n"
            f"  {official_multihop_command(directory)}\n"
            "That official script may require optional packages `pandasql` and `numpyencoder`.\n"
            f"Original error: {exc}"
        ) from exc

    missing_after = missing_multihop_result_files(directory)
    if missing_after:
        raise TimelineQAPrepError(
            "Multihop artifact generation finished, but required files are still missing:\n"
            + "\n".join(f"  {path}" for path in missing_after[:10])
            + f"\nThey should be placed in: {directory}\n"
            f"Official command: {official_multihop_command(directory)}"
        )
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
            contains_value = param_value.replace("%", "")
            filtered = filtered[filtered[column].astype(str).str.contains(contains_value, na=False, regex=False)]
        elif (" since " in question or " after " in question) and column == "year":
            filtered = filtered[filtered[column].astype(str) >= param_value]
        else:
            filtered = filtered[filtered[column].astype(str) == param_value]

    event_ids = {str(event_id) for event_id in filtered.get("eid", [])}
    dates = {normalize_date_string(date) for date in filtered.get(date_column, [])}
    evidence = []
    seen_ids: set[str] = set()
    for date in dates:
        for event_name, event in (lifelog.get(date) or {}).items():
            episode_id = str(event.get("eid"))
            if episode_id in event_ids and episode_id not in seen_ids:
                text = text_value(event.get("text_template_based") or event.get("text_model_based"))
                seen_ids.add(episode_id)
                evidence.append(
                    {
                        "episode_id": episode_id,
                        "text": f"{date}: {text}",
                        "date": date,
                        "event_name": str(event_name),
                        "event": event,
                    }
                )
    return evidence


def classify_question_type(answer_type: str, question: str) -> str:
    answer_type = (answer_type or "").strip().lower()
    question_lower = question.lower()
    if answer_type == "count":
        return "count"
    if answer_type == "average":
        return "average"
    if " first " in f" {question_lower} ":
        return "first"
    if "last" in question_lower or "most recent" in question_lower:
        return "last"
    if answer_type == "argmax":
        return "compare"
    return "general"


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


def build_multihop_candidates_for_seed(seed: int, category: str = DEFAULT_CATEGORY) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
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
                    "question_type": classify_question_type(row.get("answer_type") or "", question),
                    "_episodes": episodes,
                }
            )
    return candidates, episodes, lifelog_id


def build_multihop_candidates(
    seed: int,
    target_n: int,
    max_episodes_per_question: int | None,
    category: str = DEFAULT_CATEGORY,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    existing_records = find_existing_qa_db_records("multihop")
    if existing_records:
        record = existing_records[0]
        return existing_records, record.pop("_episodes"), record.get("lifelog_id", "qa_db")

    all_candidates: list[dict[str, Any]] = []
    all_episodes: list[dict[str, Any]] = []
    lifelog_ids: list[str] = []
    max_lifelogs = max(3, (target_n // 20) + 3)

    for offset in range(max_lifelogs):
        current_seed = seed + offset
        candidates, episodes, lifelog_id = build_multihop_candidates_for_seed(current_seed, category)
        lifelog_ids.append(lifelog_id)
        all_episodes.extend(episodes)
        all_candidates.extend(candidates)
        eligible_count = len(
            [
                candidate
                for candidate in all_candidates
                if not max_episodes_per_question
                or len(set(candidate["evidence_episode_ids"])) <= max_episodes_per_question
            ]
        )
        print(f"Collected {eligible_count} eligible multihop candidates after lifelog {lifelog_id}.")
        if eligible_count >= target_n:
            break

    return all_candidates, all_episodes, ",".join(lifelog_ids)


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
    eligible_candidates = [
        candidate
        for candidate in candidates
        if not max_episodes_per_question
        or len(set(str(episode_id) for episode_id in candidate["evidence_episode_ids"])) <= max_episodes_per_question
    ]
    if len(eligible_candidates) < n:
        raise TimelineQAPrepError(
            f"Only found {len(eligible_candidates)} eligible real {task} questions, but --n requested {n}. "
            f"Total candidates before evidence-size filtering: {len(candidates)}. "
            "Try a smaller --n, increase --max_episodes_per_question, or generate more TimelineQA lifelogs."
        )

    rng = random.Random(seed)
    selected = rng.sample(eligible_candidates, n)
    records: list[dict[str, Any]] = []
    for index, candidate in enumerate(selected):
        question_id = f"real_{task}_{index + 1:06d}"
        evidence_ids = list(dict.fromkeys(str(episode_id) for episode_id in candidate["evidence_episode_ids"]))
        candidate_episodes = candidate.get("_episodes") or all_episodes
        episodes = context_episodes(candidate_episodes, evidence_ids, max_episodes_per_question, rng)
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
            candidates, episodes, lifelog_id = build_multihop_candidates(
                args.seed,
                args.n,
                args.max_episodes_per_question,
            )

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
