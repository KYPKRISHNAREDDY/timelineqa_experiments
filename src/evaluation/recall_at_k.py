from __future__ import annotations


def recall_at_k(
    retrieved_episode_ids: list[str] | tuple[str, ...],
    evidence_episode_ids: list[str] | tuple[str, ...],
    k: int,
) -> float:
    if not evidence_episode_ids:
        return 0.0
    retrieved = {str(episode_id) for episode_id in retrieved_episode_ids[:k]}
    evidence = {str(episode_id) for episode_id in evidence_episode_ids}
    return float(bool(retrieved & evidence))
