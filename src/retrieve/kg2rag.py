"""KG²RAG-style seed, expand and rerank retrieval."""
from __future__ import annotations

import heapq
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from ..core.interfaces import Retriever
from ..core.types import Chunk, Entity, Relation, RetrievalConstraints, RetrievalResult
from .anchors import anchors_to_debug
from .registry import RetrievalContext
from .research_utils import evidence_chunk_ids, normalize_scores, relation_adjacency, semantic_scores


class KG2RAGRetriever(Retriever):
    """Semantic chunk seeds followed by bounded KG expansion and reranking."""

    name = "kg2rag"

    def __init__(self, context: RetrievalContext) -> None:
        self.context = context
        settings = context.settings
        self.default_top_k = int(settings.get("retrieval.top_k_chunks", 6))
        self.seed_chunks = int(settings.get("kg2rag.seed_chunks", 5))
        self.anchor_top_n = int(settings.get("kg2rag.anchor_top_n", 5))
        self.max_hops = int(settings.get("kg2rag.max_hops", 2))
        self.max_nodes = int(settings.get("kg2rag.max_nodes", 80))
        self.candidate_chunks = int(settings.get("kg2rag.candidate_chunks", 30))
        self.semantic_weight = float(settings.get("kg2rag.semantic_weight", 0.65))
        self.graph_weight = float(settings.get("kg2rag.graph_weight", 0.35))
        self.hop_decay = float(settings.get("kg2rag.hop_decay", 0.7))
        self._chunks = context.chunk_lookup()
        self._entities = {entity.id: entity for entity in context.store.all_entities()}
        self._relations = context.store.all_relations()

    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        year_range: Optional[Tuple[Optional[int], Optional[int]]] = None,
        constraints: Optional[RetrievalConstraints] = None,
    ) -> RetrievalResult:
        limit = top_k or self.default_top_k
        seed_hits = self.context.index.search(question, top_k=max(self.seed_chunks, limit))
        seed_chunk_ids = [chunk.id for chunk, _ in seed_hits]
        seed_vector_scores = {chunk.id: float(score) for chunk, score in seed_hits}
        anchors = self.context.anchors.resolve(question, top_n=self.anchor_top_n)

        seed_scores: Dict[str, float] = {anchor.entity.id: anchor.score for anchor in anchors}
        seed_set = set(seed_chunk_ids)
        for entity in self._entities.values():
            if any(evidence.chunk_id in seed_set for evidence in entity.evidences):
                seed_scores[entity.id] = max(seed_scores.get(entity.id, 0.0), 0.65)

        constraints = constraints or RetrievalConstraints()
        active_relations = [
            relation for relation in self._relations
            if not constraints.masks(relation.head_id, relation.type, relation.tail_id)
        ]
        adjacency = relation_adjacency(active_relations)
        node_scores, distances = self._bounded_expand(seed_scores, adjacency)
        kept_ids = set(node_scores)
        entities = [self._entities[node] for node in node_scores if node in self._entities]
        relations = [
            relation for relation in active_relations
            if relation.head_id in kept_ids and relation.tail_id in kept_ids
        ]

        candidate_ids = list(dict.fromkeys([
            *seed_chunk_ids,
            *evidence_chunk_ids(relations, entities),
        ]))[: self.candidate_chunks]
        semantic = semantic_scores(self.context.index, question, candidate_ids)
        semantic.update({
            key: max(semantic.get(key, 0.0), value)
            for key, value in seed_vector_scores.items()
        })
        graph = self._graph_chunk_scores(relations, entities, node_scores)
        semantic_norm = normalize_scores(semantic)
        graph_norm = normalize_scores(graph)
        fused = {
            chunk_id: self.semantic_weight * semantic_norm.get(chunk_id, 0.0)
            + self.graph_weight * graph_norm.get(chunk_id, 0.0)
            for chunk_id in candidate_ids
        }
        ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:limit]
        chunks = [self._chunks[chunk_id] for chunk_id, _ in ordered if chunk_id in self._chunks]

        return RetrievalResult(
            retriever_name=self.name,
            chunks=chunks,
            entities=entities,
            relations=relations,
            scores={
                **{key: round(value, 6) for key, value in node_scores.items()},
                **{key: round(value, 6) for key, value in ordered},
            },
            debug_info={
                "method": "semantic_seed_bounded_expand_graph_rerank",
                "anchors": anchors_to_debug(anchors),
                "seed_chunks": [
                    {"chunk_id": chunk.id, "doc_id": chunk.doc_id, "score": round(score, 6)}
                    for chunk, score in seed_hits
                ],
                "seed_entity_ids": sorted(seed_scores),
                "max_hops": self.max_hops,
                "max_nodes": self.max_nodes,
                "expanded_nodes": len(entities),
                "expanded_relations": len(relations),
                "hop_distance": distances,
                "candidate_chunks": len(candidate_ids),
                "masked_edge_count": len(constraints.masked_edges),
                "reranked_chunks": [
                    {
                        "chunk_id": chunk_id,
                        "score": round(score, 6),
                        "semantic": round(semantic_norm.get(chunk_id, 0.0), 6),
                        "graph": round(graph_norm.get(chunk_id, 0.0), 6),
                    }
                    for chunk_id, score in ordered
                ],
            },
        )

    def _bounded_expand(
        self, seeds: Dict[str, float], adjacency: Dict[str, list]
    ) -> Tuple[Dict[str, float], Dict[str, int]]:
        scores: Dict[str, float] = {}
        distances: Dict[str, int] = {}
        queue: List[Tuple[float, int, str]] = []
        for node, score in seeds.items():
            if node not in self._entities:
                continue
            scores[node] = max(scores.get(node, 0.0), score)
            distances[node] = 0
            heapq.heappush(queue, (-score, 0, node))

        while queue and len(scores) < self.max_nodes:
            negative_score, depth, node = heapq.heappop(queue)
            current = -negative_score
            if depth >= self.max_hops or current + 1e-12 < scores.get(node, 0.0):
                continue
            neighbors = sorted(adjacency.get(node, []), key=lambda pair: pair[0])
            for neighbor, relation in neighbors:
                if neighbor not in self._entities:
                    continue
                confidence = max((item.confidence for item in relation.evidences), default=0.5)
                propagated = current * self.hop_decay * confidence
                if propagated <= scores.get(neighbor, 0.0):
                    continue
                if neighbor not in scores and len(scores) >= self.max_nodes:
                    break
                scores[neighbor] = propagated
                distances[neighbor] = min(distances.get(neighbor, depth + 1), depth + 1)
                heapq.heappush(queue, (-propagated, depth + 1, neighbor))
        return scores, distances

    @staticmethod
    def _graph_chunk_scores(
        relations: Sequence[Relation], entities: Sequence[Entity], node_scores: Dict[str, float]
    ) -> Dict[str, float]:
        scores: Dict[str, float] = defaultdict(float)
        for relation in relations:
            support = node_scores.get(relation.head_id, 0.0) + node_scores.get(relation.tail_id, 0.0)
            for evidence in relation.evidences:
                scores[evidence.chunk_id] += support * evidence.confidence
        for entity in entities:
            for evidence in entity.evidences:
                scores[evidence.chunk_id] += 0.2 * node_scores.get(entity.id, 0.0)
        return dict(scores)
