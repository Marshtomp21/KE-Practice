"""系统内部流通的全部数据结构。

设计要点：
1. 每个 Entity / Relation 都挂载 Evidence 列表，证据从抽取阶段一路带到前端；
2. Relation 自建模起就常驻 start_year / end_year 两个字段，抽取不到时为 None，
   保证后续加入时间约束时无需改动数据结构；
3. RetrievalResult 同时容纳"文本片段"与"子图"，不为任何单一检索方式开特例字段，
   检索器私有的过程信息一律塞进 debug_info。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Evidence:
    """一条溯源记录：某个断言来自哪个文档的哪一段字符。"""

    doc_id: str
    chunk_id: str
    char_start: int
    char_end: int
    raw_text: str
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Evidence":
        return cls(
            doc_id=payload["doc_id"],
            chunk_id=payload["chunk_id"],
            char_start=int(payload["char_start"]),
            char_end=int(payload["char_end"]),
            raw_text=payload.get("raw_text", ""),
            confidence=float(payload.get("confidence", 1.0)),
        )


@dataclass
class Chunk:
    """切分后的文本片段，char_offset 指向清洗后正文中的起始位置。"""

    id: str
    doc_id: str
    text: str
    char_offset: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def char_end(self) -> int:
        return self.char_offset + len(self.text)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Chunk":
        return cls(
            id=payload["id"],
            doc_id=payload["doc_id"],
            text=payload["text"],
            char_offset=int(payload.get("char_offset", 0)),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class Entity:
    """图中的一个节点。id 由归一化阶段统一生成，name 为展示用主名。"""

    id: str
    name: str
    type: str
    aliases: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    evidences: List[Evidence] = field(default_factory=list)

    def surface_forms(self) -> List[str]:
        """主名 + 别名，去重后按长度降序，供锚点匹配做最长优先。"""
        seen: List[str] = []
        for token in [self.name, *self.aliases]:
            token = (token or "").strip()
            if token and token not in seen:
                seen.append(token)
        return sorted(seen, key=len, reverse=True)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["evidences"] = [e.to_dict() for e in self.evidences]
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Entity":
        return cls(
            id=payload["id"],
            name=payload["name"],
            type=payload["type"],
            aliases=list(payload.get("aliases", [])),
            attributes=dict(payload.get("attributes", {})),
            evidences=[Evidence.from_dict(e) for e in payload.get("evidences", [])],
        )


@dataclass
class Relation:
    """图中的一条有向边。start_year / end_year 常驻但允许为 None。"""

    id: str
    head_id: str
    tail_id: str
    type: str
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    evidences: List[Evidence] = field(default_factory=list)

    @property
    def endpoints(self) -> Tuple[str, str]:
        return self.head_id, self.tail_id

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["evidences"] = [e.to_dict() for e in self.evidences]
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Relation":
        return cls(
            id=payload["id"],
            head_id=payload["head_id"],
            tail_id=payload["tail_id"],
            type=payload["type"],
            start_year=payload.get("start_year"),
            end_year=payload.get("end_year"),
            attributes=dict(payload.get("attributes", {})),
            evidences=[Evidence.from_dict(e) for e in payload.get("evidences", [])],
        )


@dataclass
class RawDocument:
    """data/raw 下的一条原始条目。"""

    doc_id: str
    title: str
    url: str
    entity_type: str
    text: str
    infobox: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CleanedDocument:
    """清洗结果。offset_map[i] = 清洗后第 i 个字符在原文中的下标。"""

    doc_id: str
    title: str
    url: str
    entity_type: str
    text: str
    offset_map: List[int]
    infobox: Dict[str, Any] = field(default_factory=dict)

    def to_raw_span(self, start: int, end: int) -> Tuple[int, int]:
        """把清洗后正文的区间反查回原文区间。"""
        if not self.offset_map:
            return start, end
        last = len(self.offset_map) - 1
        lo = self.offset_map[min(max(start, 0), last)]
        hi = self.offset_map[min(max(end - 1, 0), last)] + 1
        return lo, max(hi, lo)


@dataclass
class RetrievalResult:
    """任何检索器的统一返回体。"""

    retriever_name: str
    chunks: List[Chunk] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    debug_info: Dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.chunks and not self.entities and not self.relations

    def to_dict(self) -> Dict[str, Any]:
        return {
            "retriever_name": self.retriever_name,
            "chunks": [c.to_dict() for c in self.chunks],
            "entities": [e.to_dict() for e in self.entities],
            "relations": [r.to_dict() for r in self.relations],
            "scores": self.scores,
            "debug_info": self.debug_info,
        }


@dataclass
class Citation:
    """答案正文引用的一条来源，前端据此高亮原文区间。"""

    marker: str
    doc_id: str
    chunk_id: str
    char_start: int
    char_end: int
    snippet: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Subgraph:
    """随答案返回给前端的子图。node_scores 决定前端节点大小。"""

    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    node_scores: Dict[str, float] = field(default_factory=dict)
    highlight_path: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "relations": [r.to_dict() for r in self.relations],
            "node_scores": self.node_scores,
            "highlight_path": self.highlight_path,
        }


@dataclass
class Answer:
    """一次问答的完整结果。"""

    text: str
    citations: List[Citation] = field(default_factory=list)
    subgraph: Subgraph = field(default_factory=Subgraph)
    retriever_name: str = ""
    latency: float = 0.0
    debug_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "citations": [c.to_dict() for c in self.citations],
            "subgraph": self.subgraph.to_dict(),
            "retriever_name": self.retriever_name,
            "latency": self.latency,
            "debug_info": self.debug_info,
        }
