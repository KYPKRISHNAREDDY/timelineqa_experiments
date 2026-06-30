#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="./.hf_cache"
export TRANSFORMERS_CACHE="./.hf_cache"

mkdir -p "$HF_HOME"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo ""
echo "Setup complete. No Hugging Face models were downloaded."
echo ""
echo "Next commands:"
echo "  bash scripts/00_clone_timelineqa.sh"
echo "  python scripts/02_make_samples.py --task atomic --n 50 --seed 42"
echo "  python scripts/04_run_model.py --task atomic --sample data/samples/atomic_n50.jsonl --model_id Qwen/Qwen2.5-0.5B-Instruct --backend hf --retriever bm25 --top_k 5 --max_new_tokens 16 --temperature 0 --output outputs/predictions/atomic_n50_qwen05b_bm25.jsonl --limit 3 --debug_first_n 3"
