"""抽取编排：逐片段跑抽取，失败降级，产物归一化后交给图构建。

降级链路：LLM 抽取 -> 单片段失败则该片段改用规则抽取 -> 整体不中断。
LLM 通道自带磁盘缓存，重跑时已处理片段不会再次调用模型（断点续跑）。
被 schema 拒绝的三元组统一写入 rejected_triples.jsonl，供抽取质量分析。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..core.config import Settings, load_schema, load_settings
from ..core.types import Chunk, CleanedDocument, Entity, Relation
from .lexicon import EntityLexicon
from .llm_extractor import LLMExtractor
from .normalizer import EntityNormalizer, NormalizationReport
from .rule_extractor import RuleExtractor


@dataclass
class ExtractionReport:
    chunks_seen: int = 0
    by_extractor: Dict[str, int] = field(default_factory=dict)
    failures: List[Dict[str, str]] = field(default_factory=list)
    rejected: List[Dict[str, object]] = field(default_factory=list)
    normalization: Optional[NormalizationReport] = None

    def note(self, extractor_name: str) -> None:
        self.by_extractor[extractor_name] = self.by_extractor.get(extractor_name, 0) + 1

    def summary(self) -> str:
        used = "，".join(f"{k} {v} 段" for k, v in sorted(self.by_extractor.items()))
        tail = self.normalization.summary() if self.normalization else "未归一化"
        return (
            f"处理片段 {self.chunks_seen}（{used or '无'}），"
            f"失败 {len(self.failures)} 段，拒绝三元组 {len(self.rejected)} 条；{tail}"
        )


class ExtractionRunner:
    """把片段流变成一份归一化后的实体与关系集合。"""

    def __init__(
        self,
        documents: Sequence[CleanedDocument],
        settings: Optional[Settings] = None,
        prefer_llm: Optional[bool] = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.schema = load_schema()
        self.documents = {d.doc_id: d for d in documents}
        self.lexicon = EntityLexicon.from_documents(documents)
        # 语料自带的链接词表（若存在）用于校验规则新建的实体名
        loaded = self.lexicon.load_known_names(
            str(self.settings.path("paths.gazetteer"))
            if self.settings.get("paths.gazetteer")
            else None
        )
        self.known_name_count = loaded
        self.rule_extractor = RuleExtractor(
            self.lexicon,
            schema=self.schema,
            context_scope=str(self.settings.get("extraction.movie_context_scope", "sentence")),
            shape_fallback=bool(self.settings.get("extraction.name_shape_fallback", True)),
        )
        self.normalizer = EntityNormalizer(self.settings)

        want_llm = (
            bool(self.settings.get("extraction.use_llm", True))
            if prefer_llm is None
            else prefer_llm
        )
        self.llm_extractor: Optional[LLMExtractor] = None
        if want_llm:
            candidate = LLMExtractor(schema=self.schema, settings=self.settings)
            if candidate.ready:
                self.llm_extractor = candidate
        self.allow_fallback = bool(self.settings.get("extraction.fallback_to_rules", True))

    def run(self, chunks: Sequence[Chunk]) -> Tuple[List[Entity], List[Relation], ExtractionReport]:
        report = ExtractionReport()
        entities: List[Entity] = []
        relations: List[Relation] = []

        for chunk in chunks:
            document = self.documents.get(chunk.doc_id)
            if document is None:
                report.failures.append({"chunk_id": chunk.id, "reason": "找不到所属文档"})
                continue
            report.chunks_seen += 1

            produced: Optional[Tuple[List[Entity], List[Relation]]] = None
            if self.llm_extractor is not None:
                try:
                    produced = self.llm_extractor.extract(chunk, document)
                    report.note(self.llm_extractor.name)
                except Exception as exc:  # 含 LLMUnavailable：任何异常都降级，不中断整体
                    report.failures.append({"chunk_id": chunk.id, "reason": f"模型抽取失败: {exc}"})
                    produced = None
            if produced is None:
                if not self.allow_fallback and self.llm_extractor is not None:
                    continue
                try:
                    produced = self.rule_extractor.extract(chunk, document)
                    report.note(self.rule_extractor.name)
                except Exception as exc:
                    report.failures.append({"chunk_id": chunk.id, "reason": f"规则抽取失败: {exc}"})
                    continue

            chunk_entities, chunk_relations = produced
            entities.extend(chunk_entities)
            relations.extend(chunk_relations)

        if self.llm_extractor is not None:
            report.rejected.extend(self.llm_extractor.rejected)

        normalized_entities, normalized_relations, norm_report = self.normalizer.normalize(
            entities, relations
        )
        report.normalization = norm_report
        self._write_reject_log(report)
        return normalized_entities, normalized_relations, report

    def _write_reject_log(self, report: ExtractionReport) -> None:
        target = self.settings.path("paths.extraction_reject_log")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for item in report.rejected:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            for item in report.failures:
                handle.write(
                    json.dumps({"kind": "failure", **item}, ensure_ascii=False) + "\n"
                )
