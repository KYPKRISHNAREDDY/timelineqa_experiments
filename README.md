# timelineqa_experiments

Clean experimental framework for TimelineQA-style question answering over timelines.

The project is designed for internship baseline runs first, then later research improvements. It runs one small model at a time, stores predictions, evaluates metrics, and combines results into a table.

## What This Project Does

1. Creates or loads TimelineQA-style samples.
2. Retrieves relevant timeline episodes with BM25.
3. Runs one local Hugging Face model through Transformers.
4. Saves every prediction as JSONL.
5. Computes Exact Match, Token F1, Denotation Accuracy, Recall@1, Recall@3, Recall@5, and average latency.

The first default experiment is:

```text
task = atomic
n = 50
model = Qwen/Qwen2.5-0.5B-Instruct
backend = hf
retriever = bm25
top_k = 5
max_new_tokens = 16
temperature = 0
```

## Folder Layout

```text
original/                 official TimelineQA repo goes here
data/raw/                 optional raw TimelineQA-style data
data/samples/             generated JSONL samples
data/indexes/             optional retrieval indexes
src/runners/              model backends
src/retrieval/            BM25 retrieval
src/evaluation/           metric functions
src/utils/                shared helpers
configs/                  model and prompt configs
scripts/                  runnable experiment scripts
notebooks/                Google Colab notebook
outputs/predictions/      prediction JSONL files
outputs/metrics/          metrics JSON files
outputs/tables/           combined CSV tables
```

## A. Local Setup

```bash
git clone https://github.com/KYPKRISHNAREDDY/timelineqa_experiments.git
cd timelineqa_experiments
bash scripts/01_setup_env.sh
```

This installs Python packages and sets local cache paths:

```bash
HF_HOME=./.hf_cache
TRANSFORMERS_CACHE=./.hf_cache
```

It does not download any Hugging Face model during setup. Models download only when you run `scripts/04_run_model.py`.

## B. Clone TimelineQA

```bash
bash scripts/00_clone_timelineqa.sh
```

The official repo is cloned into:

```text
original/TimelineQA
```

Do not edit files inside `original/TimelineQA`. Our code lives in `src/` and `scripts/`.

## C. Make Toy Sample

```bash
python scripts/02_make_samples.py --task atomic --n 50 --seed 42
```

Output:

```text
data/samples/atomic_n50.jsonl
```

This is toy data for pipeline testing only. Use it for quick smoke tests before spending GPU time.

## D. Preparing Real TimelineQA Samples

First clone the official TimelineQA repo:

```bash
bash scripts/00_clone_timelineqa.sh
```

Then prepare a real atomic sample:

```bash
python scripts/03_prepare_timelineqa_data.py --task atomic --n 50 --seed 42 --max_episodes_per_question 100 --output data/samples/real_atomic_n50.jsonl
```

This script uses the official TimelineQA generator in `original/TimelineQA` and writes converted records in this project JSONL format. It does not modify `original/TimelineQA`.

For real experiments, toy fallback is disabled by default. If real TimelineQA data cannot be generated or loaded, the script stops with a clear error. Only use `--allow_toy_fallback` for pipeline debugging.

## E. Run First Smoke Test

Run only 3 questions first:

```bash
python scripts/04_run_model.py --task atomic --sample data/samples/atomic_n50.jsonl --model_id Qwen/Qwen2.5-0.5B-Instruct --backend hf --retriever bm25 --top_k 5 --max_new_tokens 16 --temperature 0 --output outputs/predictions/atomic_n50_qwen05b_bm25.jsonl --limit 3 --debug_first_n 3
```

This downloads only the selected model. It does not run every model. The `--debug_first_n 3` flag prints the first three questions, gold answers, retrieved episode ids, retrieved context, and predictions so you can inspect retrieval quality.

To run the same smoke test on the real sample:

```bash
python scripts/04_run_model.py --task atomic --sample data/samples/real_atomic_n50.jsonl --model_id Qwen/Qwen2.5-0.5B-Instruct --backend hf --retriever bm25 --top_k 5 --max_new_tokens 16 --temperature 0 --output outputs/predictions/real_atomic_n50_qwen05b_bm25.jsonl --limit 3 --debug_first_n 3
```

To debug SmolLM2 output formatting:

```bash
python scripts/04_run_model.py \
  --task atomic \
  --sample data/samples/real_atomic_n50.jsonl \
  --model_id HuggingFaceTB/SmolLM2-1.7B-Instruct \
  --backend hf \
  --retriever bm25 \
  --top_k 5 \
  --max_new_tokens 16 \
  --temperature 0 \
  --output outputs/predictions/debug_smollm17b.jsonl \
  --limit 3 \
  --debug_first_n 3
```

## F. Evaluate

```bash
python scripts/05_evaluate.py --predictions outputs/predictions/atomic_n50_qwen05b_bm25.jsonl --task atomic --output outputs/metrics/atomic_n50_qwen05b_bm25_metrics.json
```

## G. Combine Results

```bash
python scripts/06_make_results_table.py
```

Output:

```text
outputs/tables/baseline_results.csv
```

## H. Google Colab Instructions

1. Open `notebooks/run_timelineqa_colab.ipynb`.
2. Runtime -> Change runtime type -> T4 GPU.
3. Add `HF_TOKEN` in Colab Secrets.
4. Run cells one by one.
5. First run only `--limit 3`.
6. Then run `n=50` fully.
7. Later repeat model one by one.

Recommended model order:

```text
Qwen/Qwen2.5-0.5B-Instruct
TinyLlama/TinyLlama-1.1B-Chat-v1.0
Qwen/Qwen2.5-1.5B-Instruct
HuggingFaceTB/SmolLM2-1.7B-Instruct
meta-llama/Llama-3.2-1B-Instruct
```

Note: `meta-llama/Llama-3.2-1B-Instruct` may need Hugging Face access approval. If it fails, run the other models first.

## One-Command Colab Run

The autopilot runner reads:

```text
configs/experiment_plan.yaml
```

First run a smoke test with only 3 questions per model:

```bash
python scripts/07_run_experiment_plan.py --plan configs/experiment_plan.yaml --resume --limit 3 --copy_to_drive /content/drive/MyDrive/timelineqa_results
```

If the smoke test works, run the full `n=50` baseline:

```bash
python scripts/07_run_experiment_plan.py --plan configs/experiment_plan.yaml --resume --copy_to_drive /content/drive/MyDrive/timelineqa_results
```

The script prepares the real TimelineQA sample, runs each model one by one, evaluates each prediction file, updates `outputs/tables/baseline_results.csv`, and copies results to Drive after each completed model run.

Do not start `n=500` or `n=1000` until `n=50` works. Models are still loaded one at a time internally, not all together.

## Sample JSONL Format

```json
{
  "question_id": "atomic_000001",
  "task": "atomic",
  "question": "What did I eat on 2010/01/09?",
  "gold_answer": "sushi",
  "episodes": [
    {"episode_id": "e1", "text": "2010/01/09, I had lunch. I ate sushi."},
    {"episode_id": "e2", "text": "2010/01/10, I ate pasta for dinner."}
  ],
  "evidence_episode_ids": ["e1"]
}
```

## Important Running Rule

Run one model at a time. Start small:

```text
--limit 3 first
n=50 next
n=100 later
n=500 later
n=1000 last
```

This keeps the project reproducible and avoids wasting GPU time.
