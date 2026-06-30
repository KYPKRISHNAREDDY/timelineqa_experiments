from __future__ import annotations

import os
from pathlib import Path

from src.runners.base_runner import BaseRunner


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT = """You are answering questions over a personal timeline.
Use only the provided timeline episodes.
Return ONLY the short final answer.
Do not write a sentence.
Do not explain.
Do not mention the episode id.
If the answer is a food item, return only the food item.

Question:
{question}

Timeline episodes:
{context}

Short answer:"""


class HFRunner(BaseRunner):
    """Hugging Face Transformers runner that loads exactly one model."""

    def __init__(
        self,
        model_id: str,
        max_new_tokens: int = 16,
        temperature: float = 0.0,
        prompt_template: str = DEFAULT_PROMPT,
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.prompt_template = prompt_template

        os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / ".hf_cache"))

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Missing Hugging Face dependencies. Run: pip install -r requirements.txt"
            ) from exc

        self.torch = torch
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        if not torch.cuda.is_available():
            print("WARNING: CUDA GPU was not found. The Hugging Face model will run on CPU and may be slow.")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            self.model = self._load_model(AutoModelForCausalLM, model_id, dtype)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load model '{model_id}'. If this is a gated model, request Hugging Face "
                "access and set HF_TOKEN before running. Original error: "
                f"{exc}"
            ) from exc

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()

    def _load_model(self, model_cls: object, model_id: str, dtype: object) -> object:
        common_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
        }
        try:
            return model_cls.from_pretrained(model_id, device_map="auto", **common_kwargs)
        except Exception as first_error:
            print(f"WARNING: device_map='auto' failed, retrying with a single device. Details: {first_error}")
            return model_cls.from_pretrained(model_id, **common_kwargs)

    def _build_prompt(self, question: str, context: str) -> str:
        return self.prompt_template.format(question=question, context=context)

    def run_model(self, question: str, context: str) -> str:
        prompt = self._build_prompt(question, context)
        inputs = self.tokenizer(prompt, return_tensors="pt")

        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        do_sample = self.temperature > 0
        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = self.temperature

        with self.torch.no_grad():
            output_ids = self.model.generate(**inputs, **generate_kwargs)

        prompt_length = inputs["input_ids"].shape[-1]
        answer_ids = output_ids[0][prompt_length:]
        answer = self.tokenizer.decode(answer_ids, skip_special_tokens=True)
        return answer.strip()
