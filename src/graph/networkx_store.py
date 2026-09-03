"""GraphStore 的默认实现：NetworkX MultiDiGraph + 单文件 JSON 快照。

选它是为了零外部依赖——演示现场不需要起数据库。GraphStore 仍是抽象类，
换 Neo4j 只要另写一个实现，检索层完全不用改。

内部保存的是 Entity / Relation 对象本身（挂在节点与边的 payload 属性上），
所以证据链在图里也是完整的，检索时不需要回头再查一次抽取产物。
"""
from __future__ import annotations

import json
from collections import deque
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx

from ..core.interfaces import GraphStore
from ..core.types import Entity, Relation

PAYLOAD = "payload"


def _similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left, right).ratio()


class NetworkxGraphStore(GraphStore):
    """有向多重图。同一对实体之间可以并存多种关系。"""

    def __init__(self) -> None:
        self._graph = nx.MultiDiGraph()
        self._relations: Dict[str, Relation] = {}

    # ---- 增删查 ---------------------------------------------------------

    def upsert_entity(self, entity: Entity) -> str:
        existing = self.get_entity(entity.id)
        if existing is None:
            self._graph.add_node(entity.id, **{PAYLOAD: entity})
            return entity.id

        for alias in entity.aliases:
            if alias != existing.name and alias not in existing.aliases:
                existing.aliases.append(alias)
        for key, value in entity.attributes.items():
            existing.attributes.setdefault(key, value)
        seen = {(e.chunk_id, e.char_start, e.char_end) for e in existing.evidences}
        for evidence in entity.evidences:
            token = (evidence.chunk_id, evidence.char_start, evidence.char_end)
            if token not in seen:
                seen.add(token)
                existing.evidences.append(evidence)
        return existing.id

    def upsert_relation(self, relation: Relation) -> str:
        if relation.head_id not in self._graph or relation.tail_id not in self._graph:
            raise KeyError(f"关系 {relation.id} 的端点尚未入图")

        existing = self._relations.get(relation.id)
        if existing is None:
            self._relations[relation.id] = relation
            self._graph.add_edge(
                relation.head_id, relation.tail_id, key=relation.id, **{PAYLOAD: relation}
            )
            return relation.id

        seen = {(e.chunk_id, e.char_start, e.char_end) for e in existing.evidences}
        for evidence in relation.evidences:
            token = (evidence.chunk_id, evidence.char_start, evidence.char_end)
            if token not in seen:
                seen.add(token)
                existing.evidences.append(evidence)
        if existing.start_year is None:
            existing.start_year = relation.start_year
        if existing.end_year is None:
            existing.end_year = relation.end_year
        existing.attributes.update(relation.attributes)
        return existing.id

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        node = self._graph.nodes.get(entity_id)
        return node[PAYLOAD] if node else None

    def get_relation(self, relation_id: str) -> Optional[Relation]:
        return self._relations.get(relation_id)

    def remove_entity(self, entity_id: str) -> bool:
        if entity_id not in self._graph:
            return False
        touching = [
            key
            for _, _, key in list(self._graph.in_edges(entity_id, keys=True))
            + list(self._graph.out_edges(entity_id, keys=True))
        ]
        for key in touching:
            self._relations.pop(key, None)
        self._graph.remove_node(entity_id)
        return True

    def remove_relation(self, relation_id: str) -> bool:
        relation = self._relations.pop(relation_id, None)
        if relation is None:
            return False
        if self._graph.has_edge(relation.head_id, relation.tail_id, key=relation_id):
            self._graph.remove_edge(relation.head_id, relation.tail_id, key=relation_id)
        return True

    def all_entities(self) -> List[Entity]:
        return [data[PAYLOAD] for _, data in self._graph.nodes(data=True)]

    def all_relations(self) -> List[Relation]:
        return list(self._relations.values())

    # ---- 检索支撑 -------------------------------------------------------

    def match_entities(self, text: str, limit: int = 10) -> List[Tuple[Entity, float]]:
        """先看完全命中与包含关系，再退回字符串相似度。"""
        probe = (text or "").strip()
        if not probe:
            return []
        scored: List[Tuple[Entity, float]] = []
        for entity in self.all_entities():
            best = 0.0
            for surface in entity.surface_forms():
                if surface == probe:
                    best = max(best, 1.0)
                elif surface in probe:
                    best = max(best, 0.85 + 0.1 * min(len(surface) / max(len(probe), 1), 1.0))
                elif probe in surface:
                    best = max(best, 0.7)
                else:
                    best = max(best, _similarity(surface, probe))
            if best > 0:
                scored.append((entity, min(best, 1.0)))
        scored.sort(key=lambda item: (-item[1], item[0].name))
        return scored[:limit]

    def neighborhood(
        self, seed_ids: Sequence[str], hops: int = 1, max_nodes: int = 100
    ) -> Tuple[List[Entity], List[Relation]]:
        """无向意义上的 k 跳扩展，逐层推进并在 max_nodes 处截断。"""
        present = [node for node in seed_ids if node in self._graph]
        visited: Set[str] = set(present)
        frontier = deque((node, 0) for node in present)
        picked: Set[str] = set(present)

        while frontier and len(picked) < max_nodes:
            node, depth = frontier.popleft()
            if depth >= hops:
                continue
            for neighbor in self._undirected_neighbors(node):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                picked.add(neighbor)
                frontier.append((neighbor, depth + 1))
                if len(picked) >= max_nodes:
                    break

        entities = [self.get_entity(node) for node in picked if self.get_entity(node)]
        relations = [
            relation
            for relation in self._relations.values()
            if relation.head_id in picked and relation.tail_id in picked
        ]
        return entities, relations

    def _undirected_neighbors(self, node: str):
        yield from self._graph.successors(node)
        yield from self._graph.predecessors(node)

    def as_networkx(self) -> nx.MultiDiGraph:
        return self._graph

    def subgraph_relations(self, node_ids: Sequence[str]) -> List[Relation]:
        keep = set(node_ids)
        return [
            relation
            for relation in self._relations.values()
            if relation.head_id in keep and relation.tail_id in keep
        ]

    # ---- 持久化 ---------------------------------------------------------

    def save(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        snapshot = {
            "format": "movie-graphrag-snapshot",
            "version": 1,
            "entities": [entity.to_dict() for entity in self.all_entities()],
            "relations": [relation.to_dict() for relation in self.all_relations()],
        }
        target.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")

    def load(self, path: str) -> None:
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"图快照不存在: {target}")
        snapshot = json.loads(target.read_text(encoding="utf-8"))
        self._graph = nx.MultiDiGraph()
        self._relations = {}
        for payload in snapshot.get("entities", []):
            self.upsert_entity(Entity.from_dict(payload))
        for payload in snapshot.get("relations", []):
            relation = Relation.from_dict(payload)
            if relation.head_id in self._graph and relation.tail_id in self._graph:
                self.upsert_relation(relation)

    def stats(self) -> Dict[str, Any]:
        by_entity_type: Dict[str, int] = {}
        for entity in self.all_entities():
            by_entity_type[entity.type] = by_entity_type.get(entity.type, 0) + 1
        by_relation_type: Dict[str, int] = {}
        dated = 0
        for relation in self.all_relations():
            by_relation_type[relation.type] = by_relation_type.get(relation.type, 0) + 1
            if relation.start_year is not None:
                dated += 1

        undirected = nx.Graph(self._graph)
        components = list(nx.connected_components(undirected)) if undirected.number_of_nodes() else []
        degrees = [d for _, d in undirected.degree()]
        return {
            "entities": self._graph.number_of_nodes(),
            "relations": len(self._relations),
            "entity_types": dict(sorted(by_entity_type.items(), key=lambda kv: -kv[1])),
            "relation_types": dict(sorted(by_relation_type.items(), key=lambda kv: -kv[1])),
            "relations_with_year": dated,
            "connected_components": len(components),
            "largest_component": max((len(c) for c in components), default=0),
            "average_degree": round(sum(degrees) / len(degrees), 2) if degrees else 0.0,
            "isolated_nodes": sum(1 for d in degrees if d == 0),
        }
