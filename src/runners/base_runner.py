from __future__ import annotations

from abc import ABC, abstractmethod


class BaseRunner(ABC):
    """Common interface for all model backends."""

    @abstractmethod
    def run_model(self, question: str, context: str) -> str:
        """Return only the final answer text."""
