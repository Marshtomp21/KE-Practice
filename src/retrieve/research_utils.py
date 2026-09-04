"""Shared ranking helpers for the two research-method adapters."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

from ..core.types import Chunk, Entity, Relation


def normalize_scores(values: Dict[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    peak = max(values.values())
    floor = min(values.values())
    if peak <= floor:
        return {key: 1.0 for key in values}
    return {key: (value - floor) / (peak - floor) for key, value in values.items()}


def evidence_chunk_ids(relations: Iterable[Relation], entities: Iterable[Entity]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for item in [*relations, *entities]:
        for evidence in item.evidences:
            if evidence.chunk_id not in seen:
                seen.add(evidence.chunk_id)
                ordered.append(evidence.chunk_id)
    return ordered


def semantic_scores(index, question: str, chunk_ids: Sequence[str]) -> Dict[str, float]:
    if hasattr(index, "score_chunks"):
        return dict(index.score_chunks(question, chunk_ids))
    requested = set(chunk_ids)
    hits = index.search(question, top_k=max(len(requested), 1))
    return {chunk.id: float(score) for chunk, score in hits if chunk.id in requested}


def relation_adjacency(relations: Sequence[Relation]) -> Dict[str, List[Tuple[str, Relation]]]:
    adjacency: Dict[str, List[Tuple[str, Relation]]] = defaultdict(list)
    for relation in relations:
        adjacency[relation.head_id].append((relation.tail_id, relation))
        adjacency[relation.tail_id].append((relation.head_id, relation))
    return dict(adjacency)
