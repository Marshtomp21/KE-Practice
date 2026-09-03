"""实体归一化：合并别名，拆分同名不同人。

合并策略分三档，从严到松：
1. 同类型且规范名完全一致 —— 直接合并；
2. 同类型且一方的别名覆盖另一方的名称 —— 直接合并；
3. 同类型且名称相似度过阈值，并且两者在图中至少共享一个邻居 —— 才合并。
第 3 档要求共享邻居，是为了避免把"名字长得像"的两个人错并到一起。

拆分策略（同名不同人）：同名同类型的两个实体，若其证据来自互不相交的文档集合、
在图中没有任何共享邻居、且关键属性取值冲突，则拒绝合并，让它们各自留在图中，
并把这次判定写进归一化报告，供人工复核。

合并结果全部写进 attributes["merged_from"] 与 attributes["merge_reason"]，
证据列表直接拼接，因此任何一个合并节点都能追溯回它的各个来源。
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..core.config import Settings, load_settings
from ..core.types import Entity, Relation

_STRIP_CHARS = " \t·．.，,、《》()（）\"'“”‘’"


def canonical_name(name: str) -> str:
    """比对用的规范形式：去掉首尾修饰与全部内部空白，再统一大小写。"""
    folded = "".join(ch for ch in (name or "") if not ch.isspace())
    return folded.strip(_STRIP_CHARS).lower()


def name_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


@dataclass
class NormalizationReport:
    merged: List[Dict[str, object]] = field(default_factory=list)
    split: List[Dict[str, object]] = field(default_factory=list)

    def summary(self) -> str:
        return f"合并 {len(self.merged)} 组，拆分 {len(self.split)} 组"


class EntityNormalizer:
    """把抽取阶段产生的实体集合压成一份干净的节点表，并同步改写关系端点。"""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        settings = settings or load_settings()
        self.threshold = float(settings.get("normalize.merge_threshold", 0.86))
        self.by_type_only = bool(settings.get("normalize.disambiguate_by_type", True))

    def normalize(
        self, entities: Iterable[Entity], relations: Iterable[Relation]
    ) -> Tuple[List[Entity], List[Relation], NormalizationReport]:
        entity_list = list(entities)
        relation_list = list(relations)
        report = NormalizationReport()

        neighbors = self._neighbor_index(relation_list)
        alias_index = self._build_alias_index(entity_list)
        parent: Dict[str, str] = {e.id: e.id for e in entity_list}
        by_id: Dict[str, Entity] = {e.id: e for e in entity_list}
        reasons: Dict[str, str] = {}

        def find(node: str) -> str:
            if node not in parent:
                parent[node] = node
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(left: str, right: str, reason: str) -> None:
            root_left, root_right = find(left), find(right)
            if root_left == root_right:
                return
            keeper, absorbed = sorted((root_left, root_right))
            parent[absorbed] = keeper
            reasons[absorbed] = reason

        buckets: Dict[str, List[Entity]] = defaultdict(list)
        for entity in entity_list:
            buckets[entity.type if self.by_type_only else ""].append(entity)

        for _, group in buckets.items():
            by_canonical: Dict[str, str] = {}
            for entity in group:
                key = canonical_name(entity.name)
                incumbent_id = by_canonical.get(key)
                if incumbent_id is None:
                    by_canonical[key] = entity.id
                    continue
                incumbent = by_id[incumbent_id]
                if self._looks_like_another_person(incumbent, entity, neighbors):
                    report.split.append(
                        {"name": entity.name, "type": entity.type,
                         "kept": [incumbent.id, entity.id], "reason": "属性冲突且无共同来源与邻居"}
                    )
                    continue
                union(incumbent_id, entity.id, "规范名一致")

            for entity in group:
                for alias in entity.aliases:
                    target = alias_index.get((entity.type, canonical_name(alias)))
                    if target and target != entity.id:
                        union(entity.id, target, f"别名 {alias} 指向同一实体")

            ordered = sorted(group, key=lambda e: e.name)
            for index, left in enumerate(ordered):
                for right in ordered[index + 1 : index + 12]:
                    if find(left.id) == find(right.id):
                        continue
                    score = name_similarity(canonical_name(left.name), canonical_name(right.name))
                    if score < self.threshold:
                        continue
                    if neighbors[left.id] & neighbors[right.id]:
                        union(left.id, right.id, f"名称相似度 {score:.2f} 且共享邻居")

        merged_entities = self._collapse(entity_list, parent, find, reasons, report)
        rewritten = self._rewrite_relations(relation_list, find)
        return merged_entities, rewritten, report

    # ---- 内部工具 -------------------------------------------------------

    def _neighbor_index(self, relations: Sequence[Relation]) -> Dict[str, Set[str]]:
        index: Dict[str, Set[str]] = defaultdict(set)
        for relation in relations:
            index[relation.head_id].add(relation.tail_id)
            index[relation.tail_id].add(relation.head_id)
        return index

    def _build_alias_index(self, entities: Sequence[Entity]) -> Dict[Tuple[str, str], str]:
        index: Dict[Tuple[str, str], str] = {}
        for entity in entities:
            index.setdefault((entity.type, canonical_name(entity.name)), entity.id)
        return index

    def _collapse(
        self,
        entities: Sequence[Entity],
        parent: Dict[str, str],
        find,
        reasons: Dict[str, str],
        report: NormalizationReport,
    ) -> List[Entity]:
        groups: Dict[str, List[Entity]] = defaultdict(list)
        for entity in entities:
            groups[find(entity.id)].append(entity)

        result: List[Entity] = []
        for root, members in groups.items():
            # 主名优先取证据最多的写法；同等时偏好不含空白、更完整的写法
            members.sort(
                key=lambda e: (
                    -len(e.evidences),
                    any(ch.isspace() for ch in e.name),
                    -len(e.name),
                    e.name,
                )
            )
            primary = members[0]
            merged = Entity(
                id=root,
                name=primary.name,
                type=primary.type,
                aliases=list(primary.aliases),
                attributes=dict(primary.attributes),
                evidences=list(primary.evidences),
            )
            for other in members[1:]:
                if other.name != merged.name and other.name not in merged.aliases:
                    merged.aliases.append(other.name)
                for alias in other.aliases:
                    if alias != merged.name and alias not in merged.aliases:
                        merged.aliases.append(alias)
                for key, value in other.attributes.items():
                    merged.attributes.setdefault(key, value)
                merged.evidences.extend(other.evidences)
            if len(members) > 1:
                merged.attributes["merged_from"] = [m.id for m in members]
                merged.attributes["merge_reason"] = [
                    reasons.get(m.id, "同组") for m in members[1:]
                ]
                report.merged.append(
                    {"kept": merged.id, "absorbed": [m.id for m in members[1:]]}
                )
            result.append(merged)
        return result

    def _rewrite_relations(self, relations: Sequence[Relation], find) -> List[Relation]:
        collapsed: Dict[str, Relation] = {}
        for relation in relations:
            head = find(relation.head_id)
            tail = find(relation.tail_id)
            if head == tail:
                continue  # 合并后自环没有信息量
            new_id = f"{head}|{relation.type}|{tail}"
            existing = collapsed.get(new_id)
            if existing is None:
                collapsed[new_id] = Relation(
                    id=new_id,
                    head_id=head,
                    tail_id=tail,
                    type=relation.type,
                    start_year=relation.start_year,
                    end_year=relation.end_year,
                    attributes=dict(relation.attributes),
                    evidences=list(relation.evidences),
                )
                continue
            existing.evidences.extend(relation.evidences)
            if existing.start_year is None:
                existing.start_year = relation.start_year
            if existing.end_year is None:
                existing.end_year = relation.end_year
        return list(collapsed.values())

    def _looks_like_another_person(
        self, left: Entity, right: Entity, neighbors: Dict[str, Set[str]]
    ) -> bool:
        """同名不同人的判据：来源文档不相交、图上无共同邻居、且关键属性取值冲突。"""
        shared_docs = {e.doc_id for e in left.evidences} & {e.doc_id for e in right.evidences}
        if shared_docs or (neighbors[left.id] & neighbors[right.id]):
            return False
        ignored = {"merged_from", "merge_reason", "extractor"}
        return any(
            key not in ignored and key in right.attributes and right.attributes[key] != value
            for key, value in left.attributes.items()
        )
