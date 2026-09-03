"""答案生成。

两个实现共用 AnswerGenerator 接口：
- LLMAnswerGenerator：把组装好的上下文交给模型，要求逐句标注引用编号；
- StructuredAnswerGenerator：不依赖任何外部服务，直接把子图与片段整理成
  一段带引用的结构化回答。它既是模型不可用时的降级通道，也是对比实验里
  排除"生成模型发挥波动"这一干扰项的手段。

两者都遵守同一条纪律：检索为空时明确说查无此内容，不编造。
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

from ..core.config import Settings, load_schema, load_settings, read_prompt
from ..core.interfaces import AnswerGenerator
from ..core.llm import ChatClient, LLMUnavailable
from ..core.types import Answer, Citation, Entity, Relation, RetrievalResult, Subgraph
from .context_builder import BuiltContext, ContextBuilder

EMPTY_REPLY = "根据现有语料与图谱，没有检索到与该问题相关的内容，因此无法作答。"


def _subgraph_of(result: RetrievalResult) -> Subgraph:
    node_scores = {
        entity.id: float(result.scores.get(entity.id, 0.0)) for entity in result.entities
    }
    highlight = [
        entity_id
        for entity_id, _ in sorted(node_scores.items(), key=lambda kv: -kv[1])[:6]
    ]
    return Subgraph(
        entities=list(result.entities),
        relations=list(result.relations),
        node_scores=node_scores,
        highlight_path=highlight,
    )


class StructuredAnswerGenerator(AnswerGenerator):
    """确定性生成：按关系类型归并子图，逐条给出引用。"""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or load_settings()
        self.schema = load_schema()
        self.builder = ContextBuilder(self.settings)

    def generate(self, question: str, result: RetrievalResult) -> Answer:
        started = time.perf_counter()
        context = self.builder.build(result)
        if context.is_empty:
            return Answer(
                text=EMPTY_REPLY,
                citations=[],
                subgraph=_subgraph_of(result),
                retriever_name=result.retriever_name,
                latency=time.perf_counter() - started,
                debug_info={"generator": "structured", "reason": "检索结果为空"},
            )

        names = {entity.id: entity.name for entity in result.entities}
        anchors = self._anchor_ids(result)
        lines: List[str] = []

        if result.relations:
            focus = self._relations_touching(result.relations, anchors) or result.relations
            grouped: Dict[str, List[Relation]] = {}
            for relation in focus:
                grouped.setdefault(relation.type, []).append(relation)

            for relation_type, group in sorted(grouped.items(), key=lambda kv: -len(kv[1])):
                spec = self.schema.relation_spec(relation_type)
                label = spec.label if spec else relation_type
                rendered = []
                for relation in group[:12]:
                    head = names.get(relation.head_id, relation.head_id)
                    tail = names.get(relation.tail_id, relation.tail_id)
                    when = f"（{relation.start_year}）" if relation.start_year else ""
                    marks = self._marks_for(relation, context)
                    rendered.append(f"{head} → {tail}{when}{marks}")
                more = f"，另有 {len(group) - 12} 条同类关系" if len(group) > 12 else ""
                lines.append(f"【{label}】共 {len(group)} 条：" + "；".join(rendered) + more + "。")

            if len(anchors) >= 2:
                lines.append(self._pair_verdict(anchors, result, names))
        else:
            lines.append(
                "本次检索只命中了文本片段，没有命中任何图谱关系，以下结论仅来自原文。"
            )

        for citation in context.citations[:3]:
            lines.append(f"[{citation.marker}] 原文：{citation.snippet[:120]}")

        return Answer(
            text="\n".join(lines),
            citations=context.citations,
            subgraph=_subgraph_of(result),
            retriever_name=result.retriever_name,
            latency=time.perf_counter() - started,
            debug_info={
                "generator": "structured",
                "relation_count": len(result.relations),
                "chunk_count": len(context.used_chunks),
                "context_truncated": context.truncated,
            },
        )

    # ---- 内部工具 -------------------------------------------------------

    def _anchor_ids(self, result: RetrievalResult) -> List[str]:
        """按名称去重：同名的 Person 与 Character 会同时成为锚点，
        若不去重，两两比较就会退化成拿一个名字和它自己比。"""
        debug = result.debug_info
        anchors = debug.get("anchors") or debug.get("graph_debug", {}).get("anchors") or []
        picked: List[str] = []
        seen_names: set[str] = set()
        for item in anchors:
            if not isinstance(item, dict):
                continue
            name = item.get("name", "")
            if name in seen_names:
                continue
            seen_names.add(name)
            picked.append(item["entity_id"])
        return picked

    def _relations_touching(
        self, relations: Sequence[Relation], anchors: Sequence[str]
    ) -> List[Relation]:
        if not anchors:
            return []
        keep = set(anchors)
        return [r for r in relations if r.head_id in keep or r.tail_id in keep]

    def _marks_for(self, relation: Relation, context: BuiltContext) -> str:
        chunk_ids = {c.id: c for c in context.used_chunks}
        marks = []
        for citation in context.citations:
            if any(e.chunk_id == citation.chunk_id for e in relation.evidences):
                marks.append(f"[{citation.marker}]")
        return "".join(marks[:2])

    def _pair_verdict(
        self, anchors: Sequence[str], result: RetrievalResult, names: Dict[str, str]
    ) -> str:
        """两个锚点之间有没有直接关系——反事实否定题靠这句话给出明确结论。"""
        left, right = anchors[0], anchors[1]
        direct = [
            r
            for r in result.relations
            if {r.head_id, r.tail_id} == {left, right}
        ]
        left_name = names.get(left, left)
        right_name = names.get(right, right)
        if direct:
            spec = self.schema.relation_spec(direct[0].type)
            label = spec.label if spec else direct[0].type
            return f"结论：{left_name} 与 {right_name} 之间存在「{label}」关系。"

        bridges = self._bridges(left, right, result.relations)
        if bridges:
            rendered = "、".join(names.get(b, b) for b in bridges[:5])
            return (
                f"结论：图谱中 {left_name} 与 {right_name} 之间不存在直接关系，"
                f"但两者通过 {rendered} 产生关联。"
            )
        return (
            f"结论：图谱中查不到 {left_name} 与 {right_name} 之间的任何直接关系，"
            f"在本次检索到的子图内也没有把两者连起来的中间实体。"
        )

    def _bridges(self, left: str, right: str, relations: Sequence[Relation]) -> List[str]:
        adjacency: Dict[str, set] = {}
        for relation in relations:
            adjacency.setdefault(relation.head_id, set()).add(relation.tail_id)
            adjacency.setdefault(relation.tail_id, set()).add(relation.head_id)
        return sorted(adjacency.get(left, set()) & adjacency.get(right, set()))


class LLMAnswerGenerator(AnswerGenerator):
    """走大模型生成；调用失败时自动退回结构化生成，保证接口永远有返回。"""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Optional[ChatClient] = None,
        fallback: Optional[AnswerGenerator] = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.client = client or ChatClient.from_settings(self.settings)
        self.builder = ContextBuilder(self.settings)
        self.fallback = fallback or StructuredAnswerGenerator(self.settings)
        self._system_prompt = read_prompt("answer_system.txt")
        self._user_template = read_prompt("answer_user.txt")

    @property
    def ready(self) -> bool:
        return self.client.ready

    def generate(self, question: str, result: RetrievalResult) -> Answer:
        started = time.perf_counter()
        context = self.builder.build(result)
        if context.is_empty and bool(self.settings.get("generation.refuse_when_empty", True)):
            return Answer(
                text=EMPTY_REPLY,
                citations=[],
                subgraph=_subgraph_of(result),
                retriever_name=result.retriever_name,
                latency=time.perf_counter() - started,
                debug_info={"generator": "llm", "reason": "检索结果为空，直接拒答"},
            )

        prompt = self._user_template.format(
            question=question, context_block=context.as_prompt_block()
        )
        try:
            text = self.client.complete(
                self._system_prompt,
                prompt,
                temperature=float(self.settings.get("generation.temperature", 0.0)),
                max_tokens=int(self.settings.get("generation.max_tokens", 1200)),
            ).strip()
        except LLMUnavailable as exc:
            answer = self.fallback.generate(question, result)
            answer.debug_info["generator"] = "structured(降级)"
            answer.debug_info["llm_error"] = str(exc)
            return answer

        return Answer(
            text=text,
            citations=context.citations,
            subgraph=_subgraph_of(result),
            retriever_name=result.retriever_name,
            latency=time.perf_counter() - started,
            debug_info={
                "generator": "llm",
                "model": self.client.model,
                "context_chars": len(prompt),
                "context_truncated": context.truncated,
            },
        )


def build_generator(settings: Optional[Settings] = None) -> AnswerGenerator:
    """配置齐全就用模型，否则用结构化生成。调用方不需要知道用的是哪个。"""
    settings = settings or load_settings()
    candidate = LLMAnswerGenerator(settings)
    return candidate if candidate.ready else StructuredAnswerGenerator(settings)
