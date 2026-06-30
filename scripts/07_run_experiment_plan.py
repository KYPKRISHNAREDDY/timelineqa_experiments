from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import ensure_dir, read_yaml


def as_list(value: Any, field_name: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Plan field '{field_name}' must be a non-empty list.")
    return value


def require_plan_fields(plan: dict[str, Any]) -> None:
    required = [
        "task",
        "data_source",
        "seed",
        "sample_sizes",
        "max_episodes_per_question",
        "retriever",
        "top_k",
        "max_new_tokens",
        "temperature",
        "backend",
        "models",
    ]
    missing = [field for field in required if field not in plan]
    if missing:
        raise ValueError(f"Missing required plan fields: {', '.join(missing)}")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


class Logger:
    def __init__(self, log_path: Path):
        ensure_dir(log_path.parent)
        self.log_path = log_path
        self.handle = log_path.open("a", encoding="utf-8")

    def close(self) -> None:
        self.handle.close()

    def log(self, message: str = "") -> None:
        print(message)
        self.handle.write(message + "\n")
        self.handle.flush()


def run_command(command: list[str], logger: Logger, dry_run: bool) -> None:
    logger.log(f"$ {command_text(command)}")
    if dry_run:
        logger.log("DRY RUN: command not executed.")
        return

    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        logger.log(line.rstrip())
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {command_text(command)}")


def sample_path(plan: dict[str, Any], n: int) -> Path:
    task = plan["task"]
    data_source = plan["data_source"]
    if data_source == "real_timelineqa":
        return PROJECT_ROOT / "data" / "samples" / f"real_{task}_n{n}.jsonl"
    if data_source == "toy":
        return PROJECT_ROOT / "data" / "samples" / f"{task}_n{n}.jsonl"
    raise ValueError(f"Unsupported data_source: {data_source}")


def prepare_command(plan: dict[str, Any], n: int, output_path: Path) -> list[str]:
    task = plan["task"]
    data_source = plan["data_source"]
    if data_source == "real_timelineqa":
        command = [
            sys.executable,
            "scripts/03_prepare_timelineqa_data.py",
            "--task",
            task,
            "--n",
            str(n),
            "--seed",
            str(plan["seed"]),
            "--max_episodes_per_question",
            str(plan["max_episodes_per_question"]),
            "--output",
            rel(output_path),
        ]
        if task == "multihop" and "max_seed_attempts" in plan:
            command.extend(["--max_seed_attempts", str(plan["max_seed_attempts"])])
        return command
    if data_source == "toy":
        return [
            sys.executable,
            "scripts/02_make_samples.py",
            "--task",
            task,
            "--n",
            str(n),
            "--seed",
            str(plan["seed"]),
            "--output",
            rel(output_path),
        ]
    raise ValueError(f"Unsupported data_source: {data_source}")


def prediction_path(plan: dict[str, Any], n: int, short_name: str) -> Path:
    task = plan["task"]
    retriever = plan["retriever"]
    prefix = "real" if plan["data_source"] == "real_timelineqa" else str(plan["data_source"])
    return PROJECT_ROOT / "outputs" / "predictions" / f"{prefix}_{task}_n{n}_{short_name}_{retriever}.jsonl"


def metrics_path(plan: dict[str, Any], n: int, short_name: str) -> Path:
    task = plan["task"]
    retriever = plan["retriever"]
    prefix = "real" if plan["data_source"] == "real_timelineqa" else str(plan["data_source"])
    return PROJECT_ROOT / "outputs" / "metrics" / f"{prefix}_{task}_n{n}_{short_name}_{retriever}_metrics.json"


def model_command(
    plan: dict[str, Any],
    n: int,
    model: dict[str, Any],
    input_sample: Path,
    output_predictions: Path,
    limit: int | None,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/04_run_model.py",
        "--task",
        str(plan["task"]),
        "--sample",
        rel(input_sample),
        "--model_id",
        str(model["model_id"]),
        "--backend",
        str(plan["backend"]),
        "--retriever",
        str(plan["retriever"]),
        "--top_k",
        str(plan["top_k"]),
        "--max_new_tokens",
        str(plan["max_new_tokens"]),
        "--temperature",
        str(plan["temperature"]),
        "--output",
        rel(output_predictions),
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    return command


def evaluate_command(plan: dict[str, Any], predictions: Path, metrics: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/05_evaluate.py",
        "--predictions",
        rel(predictions),
        "--task",
        str(plan["task"]),
        "--output",
        rel(metrics),
    ]


def table_command() -> list[str]:
    return [sys.executable, "scripts/06_make_results_table.py"]


def copy_results(copy_to_drive: str | None, logger: Logger, dry_run: bool) -> None:
    if not copy_to_drive:
        return

    destination = Path(copy_to_drive)
    logger.log(f"Copying outputs/ and data/samples/ to {destination}")
    if dry_run:
        logger.log("DRY RUN: copy not executed.")
        return

    ensure_dir(destination)
    shutil.copytree(PROJECT_ROOT / "outputs", destination / "outputs", dirs_exist_ok=True)
    ensure_dir(destination / "data")
    shutil.copytree(PROJECT_ROOT / "data" / "samples", destination / "data" / "samples", dirs_exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a TimelineQA experiment plan one model at a time.")
    parser.add_argument("--plan", default="configs/experiment_plan.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--copy_to_drive", default=None)
    args = parser.parse_args()

    plan_path = Path(args.plan)
    if not plan_path.is_absolute():
        plan_path = PROJECT_ROOT / plan_path
    plan = read_yaml(plan_path)
    if not isinstance(plan, dict):
        raise ValueError(f"Plan file must contain a YAML mapping: {plan_path}")
    require_plan_fields(plan)

    sample_sizes = [int(value) for value in as_list(plan["sample_sizes"], "sample_sizes")]
    models = as_list(plan["models"], "models")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = PROJECT_ROOT / "outputs" / "logs" / f"experiment_plan_{timestamp}.log"
    logger = Logger(log_path)

    try:
        logger.log(f"Experiment plan: {rel(plan_path)}")
        logger.log(f"Log file: {rel(log_path)}")
        logger.log(f"Dry run: {args.dry_run}")
        logger.log(f"Resume: {args.resume}")
        if args.limit is not None:
            logger.log(f"Limit forwarded to model runs: {args.limit}")
        logger.log("")

        for n in sample_sizes:
            prepared_sample = sample_path(plan, n)
            logger.log(f"[1/4] Preparing data for n={n}...")
            run_command(prepare_command(plan, n, prepared_sample), logger, args.dry_run)
            logger.log("")

            for model_index, model in enumerate(models, start=1):
                short_name = str(model.get("short_name") or model.get("model_id", f"model{model_index}"))
                model_id = str(model["model_id"])
                predictions = prediction_path(plan, n, short_name)
                metrics = metrics_path(plan, n, short_name)

                run_label = f"n={n}, model={short_name}, model_id={model_id}"
                if args.resume and metrics.exists():
                    logger.log(f"SKIPPING existing completed run: {run_label}")
                    logger.log("")
                    continue

                logger.log(f"[2/4] Running model ({model_index}/{len(models)}): {run_label}")
                run_command(model_command(plan, n, model, prepared_sample, predictions, args.limit), logger, args.dry_run)
                logger.log("")

                logger.log(f"[3/4] Evaluating: {run_label}")
                run_command(evaluate_command(plan, predictions, metrics), logger, args.dry_run)
                copy_results(args.copy_to_drive, logger, args.dry_run)
                logger.log("")

        logger.log("[4/4] Updating results table...")
        run_command(table_command(), logger, args.dry_run)
        logger.log("")
        logger.log("Experiment plan finished.")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
