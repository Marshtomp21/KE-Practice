"""锚点识别：把自然语言问题落到图上的具体节点。

两级策略：
1. 精确扫描——用图中全部实体的主名与别名拼成一个长词优先的正则，直接在问句里
   找命中。中文没有空格，这一步比分词更可靠。
2. 模糊兜底——一个精确命中都没有时，退回 GraphStore 的模糊匹配，取过阈值的若干个。

三种用到图的检索器共用它，保证它们的入口条件完全一致，对比实验才公平。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..core.interfaces import GraphStore
from ..core.types import Entity


@dataclass
class Anchor:
    entity: Entity
    score: float
    matched_text: str
    how: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "entity_id": self.entity.id,
            "name": self.entity.name,
            "type": self.entity.type,
            "score": round(self.score, 4),
            "matched_text": self.matched_text,
            "how": self.how,
        }


class AnchorResolver:
    """在一份实体表上做一次性建索引，之后每个问题只扫一遍正则。"""

    def __init__(self, store: GraphStore, min_score: float = 0.55) -> None:
        self.store = store
        self.min_score = min_score
        self._surface_to_entities: Dict[str, List[Entity]] = {}
        self._pattern: Optional[re.Pattern[str]] = None
        self._rebuild()

    def _rebuild(self) -> None:
        table: Dict[str, List[Entity]] = {}
        for entity in self.store.all_entities():
            for surface in entity.surface_forms():
                if len(surface) < 2:
                    continue
                table.setdefault(surface, []).append(entity)
        self._surface_to_entities = table
        if table:
            ordered = sorted(table, key=len, reverse=True)
            self._pattern = re.compile("|".join(re.escape(s) for s in ordered))
        else:
            self._pattern = None

    def resolve(self, question: str, top_n: int = 5) -> List[Anchor]:
        anchors: List[Anchor] = []
        seen: set[str] = set()

        if self._pattern is not None:
            covered: List[Tuple[int, int]] = []
            for match in self._pattern.finditer(question or ""):
                span = (match.start(), match.end())
                if any(span[0] >= lo and span[1] <= hi for lo, hi in covered):
                    continue
                covered.append(span)
                for entity in self._surface_to_entities.get(match.group(), []):
                    if entity.id in seen:
                        continue
                    seen.add(entity.id)
                    anchors.append(
                        Anchor(entity=entity, score=1.0, matched_text=match.group(), how="精确命中")
                    )

        if not anchors:
            for entity, score in self.store.match_entities(question, limit=top_n * 2):
                if score < self.min_score or entity.id in seen:
                    continue
                seen.add(entity.id)
                anchors.append(
                    Anchor(entity=entity, score=score, matched_text=entity.name, how="模糊匹配")
                )

        # 长名称信息量更大，同分时优先
        anchors.sort(key=lambda a: (-a.score, -len(a.matched_text)))
        return anchors[:top_n]


def anchors_to_debug(anchors: Sequence[Anchor]) -> List[Dict[str, object]]:
    return [anchor.to_dict() for anchor in anchors]
