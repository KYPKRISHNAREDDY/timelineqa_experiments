from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ModelOutput:
    """Raw and lightly cleaned text returned by a model backend."""

    raw_generated_text: str
    cleaned_answer: str


class BaseRunner(ABC):
    """Common interface for all model backends."""

    @abstractmethod
    def run_model(self, question: str, context: str) -> ModelOutput:
        """Return raw generated text plus the backend's basic cleaned answer."""
