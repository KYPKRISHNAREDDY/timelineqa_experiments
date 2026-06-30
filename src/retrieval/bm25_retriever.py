from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Okapi


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_ID_KEYS = ("episode_id", "id", "uid", "episodeId", "episodeID")
_TEXT_KEYS = ("text", "episode_text", "content", "description", "event", "passage", "summary")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _first_present(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


class BM25Retriever:
    """Small BM25 wrapper for TimelineQA episode retrieval."""

    def __init__(self, episodes: list[dict[str, Any]]):
        if not episodes:
            raise ValueError("BM25Retriever needs at least one episode.")

        self.episodes = [self._normalize_episode(episode, index) for index, episode in enumerate(episodes)]
        tokenized = [_tokenize(episode["text"]) for episode in self.episodes]
        self._bm25 = BM25Okapi(tokenized)

    @staticmethod
    def _normalize_episode(episode: dict[str, Any], index: int) -> dict[str, str]:
        episode_id = _first_present(episode, _ID_KEYS)
        text = _first_present(episode, _TEXT_KEYS)

        if text is None:
            text = " ".join(str(value) for value in episode.values() if value is not None)

        return {
            "episode_id": str(episode_id if episode_id is not None else f"episode_{index:06d}"),
            "text": str(text),
        }

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            selected = self.episodes[:top_k]
            return [{**episode, "score": 0.0} for episode in selected]

        scores = self._bm25.get_scores(query_tokens)
        ranked_indexes = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)

        results: list[dict[str, Any]] = []
        for index in ranked_indexes[:top_k]:
            episode = self.episodes[index]
            results.append(
                {
                    "episode_id": episode["episode_id"],
                    "text": episode["text"],
                    "score": float(scores[index]),
                }
            )
        return results
