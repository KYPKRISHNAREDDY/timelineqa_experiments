from __future__ import annotations

import shutil
import subprocess

from src.runners.base_runner import BaseRunner, ModelOutput


DEFAULT_PROMPT = """You are answering questions over a personal timeline.
Use only the provided timeline episodes.
Return only the final answer.
Do not explain.

Question:
{question}

Timeline episodes:
{context}

Answer:"""


class OllamaRunner(BaseRunner):
    """Optional local Ollama backend."""

    def __init__(
        self,
        model_id: str,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        prompt_template: str = DEFAULT_PROMPT,
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.prompt_template = prompt_template

        if shutil.which("ollama") is None:
            raise RuntimeError("Ollama was not found. Install Ollama and make sure the 'ollama' command is on PATH.")

    def run_model(self, question: str, context: str) -> ModelOutput:
        prompt = self.prompt_template.format(question=question, context=context)
        try:
            result = subprocess.run(
                ["ollama", "run", self.model_id, prompt],
                capture_output=True,
                check=True,
                text=True,
                encoding="utf-8",
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Ollama failed. Make sure the Ollama app/server is running and the model is pulled. "
                f"Original error: {exc.stderr.strip()}"
            ) from exc

        raw_generated_text = result.stdout
        return ModelOutput(
            raw_generated_text=raw_generated_text,
            cleaned_answer=raw_generated_text.strip(),
        )
