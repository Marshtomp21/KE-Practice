"""上下文组装：把检索结果压成一段有编号、可引用、长度受控的提示文本。

编号规则是全链路引用的锚：文本片段编号为 [S1] [S2]…，三元组编号为 [G1] [G2]…。
生成阶段要求答案句尾必须带上这些编号，前端再按编号把原文区间高亮出来。
截断在片段边界上进行，不会把一段话截一半，避免引用指向不完整的原文。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..core.config import Settings, load_schema, load_settings
from ..core.types import Chunk, Citation, Entity, Relation, RetrievalResult


@dataclass
class BuiltContext:
    passage_block: str
    triple_block: str
    citations: List[Citation] = field(default_factory=list)
    used_chunks: List[Chunk] = field(default_factory=list)
    used_relations: List[Relation] = field(default_factory=list)
    truncated: bool = False

    @property
    def is_empty(self) -> bool:
        return not self.used_chunks and not self.used_relations

    def as_prompt_block(self) -> str:
        parts = []
        if self.triple_block:
            parts.append("图谱事实：\n" + self.triple_block)
        if self.passage_block:
            parts.append("原文片段：\n" + self.passage_block)
        return "\n\n".join(parts) if parts else "（检索没有返回任何内容）"


class ContextBuilder:
    """检索结果 -> 提示上下文 + 引用清单。"""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or load_settings()
        self.schema = load_schema()
        self.max_chars = int(self.settings.get("retrieval.max_context_chars", 6000))

    def build(self, result: RetrievalResult) -> BuiltContext:
        names = {entity.id: entity.name for entity in result.entities}
        triple_lines: List[str] = []
        used_relations: List[Relation] = []

        ordered_relations = sorted(
            result.relations,
            key=lambda r: -(result.scores.get(r.head_id, 0.0) + result.scores.get(r.tail_id, 0.0)),
        )
        for relation in ordered_relations:
            head = names.get(relation.head_id)
            tail = names.get(relation.tail_id)
            if not head or not tail:
                continue
            spec = self.schema.relation_spec(relation.type)
            label = spec.label if spec else relation.type
            when = f"（{relation.start_year} 年）" if relation.start_year else ""
            marker = f"G{len(triple_lines) + 1}"
            triple_lines.append(f"[{marker}] {head} —{label}{when}→ {tail}")
            used_relations.append(relation)

        budget = self.max_chars
        triple_block = ""
        for line in triple_lines:
            if len(triple_block) + len(line) + 1 > budget // 2:
                break
            triple_block += line + "\n"
        triple_block = triple_block.rstrip()
        budget -= len(triple_block)

        passage_lines: List[str] = []
        citations: List[Citation] = []
        used_chunks: List[Chunk] = []
        truncated = False
        for chunk in result.chunks:
            marker = f"S{len(used_chunks) + 1}"
            title = chunk.metadata.get("title", chunk.doc_id)
            block = f"[{marker}]（{title}）{chunk.text}"
            if len(block) > budget:
                truncated = True
                break
            budget -= len(block) + 1
            passage_lines.append(block)
            used_chunks.append(chunk)
            citations.append(
                Citation(
                    marker=marker,
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.id,
                    char_start=chunk.char_offset,
                    char_end=chunk.char_end,
                    snippet=chunk.text,
                )
            )

        return BuiltContext(
            passage_block="\n".join(passage_lines),
            triple_block=triple_block,
            citations=citations,
            used_chunks=used_chunks,
            used_relations=used_relations[: triple_block.count("\n") + 1 if triple_block else 0],
            truncated=truncated,
        )
