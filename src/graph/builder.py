"""图构建：把归一化后的实体与关系灌进 GraphStore，并落盘快照。

端点缺失的关系会被丢弃而不是抛异常——抽取阶段偶发的悬空引用不应该让整条
流水线停下，丢弃记录进 report 里，后续用它做抽取质量分析。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..core.config import Settings, load_settings
from ..core.interfaces import GraphStore
from ..core.types import Entity, Relation
from .networkx_store import NetworkxGraphStore


@dataclass
class BuildReport:
    entities_written: int = 0
    relations_written: int = 0
    dangling: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"入图实体 {self.entities_written}，关系 {self.relations_written}，"
            f"丢弃悬空关系 {len(self.dangling)}"
        )


class GraphBuilder:
    """默认装配 NetworkX 后端；传入别的 GraphStore 即可换后端。"""

    def __init__(
        self, store: Optional[GraphStore] = None, settings: Optional[Settings] = None
    ) -> None:
        self.settings = settings or load_settings()
        self.store = store or NetworkxGraphStore()

    def build(
        self, entities: Sequence[Entity], relations: Sequence[Relation]
    ) -> BuildReport:
        report = BuildReport()
        known = set()
        for entity in entities:
            self.store.upsert_entity(entity)
            known.add(entity.id)
            report.entities_written += 1

        for relation in relations:
            if relation.head_id not in known or relation.tail_id not in known:
                report.dangling.append(relation.id)
                continue
            self.store.upsert_relation(relation)
            report.relations_written += 1

        report.stats = self.store.stats()
        return report

    def persist(self) -> str:
        target = self.settings.path("paths.graph_file")
        self.store.save(str(target))
        return str(target)


def load_graph(settings: Optional[Settings] = None) -> NetworkxGraphStore:
    """从快照恢复图。检索层与 API 层都走这个入口。"""
    settings = settings or load_settings()
    store = NetworkxGraphStore()
    store.load(str(settings.path("paths.graph_file")))
    return store
