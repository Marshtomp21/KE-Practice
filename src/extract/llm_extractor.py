"""LLM 抽取器。

产出必须经 SchemaRegistry 校验，不合法的三元组直接丢弃并交由调用方记账。
每个片段的模型原始返回落盘缓存，重跑时命中缓存即跳过，满足断点续跑要求。
模型不可用时抛出 LLMUnavailable，由 ExtractionRunner 决定是否降级到规则通道。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.config import SchemaRegistry, Settings, load_schema, load_settings, read_prompt
from ..core.interfaces import TripleExtractor
from ..core.llm import ChatClient, LLMUnavailable, parse_json_block
from ..core.types import Chunk, CleanedDocument, Entity, Evidence, Relation
from .rule_extractor import entity_key


def _year_or_none(value: Any) -> Optional[int]:
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None
    return year if 1800 <= year <= 2100 else None


class LLMExtractor(TripleExtractor):
    """把一个片段交给模型，换回受本体约束的实体与关系。"""

    name = "llm"

    def __init__(
        self,
        client: Optional[ChatClient] = None,
        schema: Optional[SchemaRegistry] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.schema = schema or load_schema()
        self.client = client or ChatClient.from_settings(self.settings)
        self.cache_dir: Path = self.settings.path("paths.extraction_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_confidence = float(self.settings.get("extraction.min_confidence", 0.35))
        self.max_chars = int(self.settings.get("extraction.max_chars_per_call", 1800))
        self.rejected: List[Dict[str, Any]] = []
        self._system_prompt = read_prompt("extraction_system.txt")
        self._user_template = read_prompt("extraction_user.txt")

    @property
    def ready(self) -> bool:
        return self.client.ready

    # ---- 缓存 -----------------------------------------------------------

    def _cache_file(self, chunk: Chunk) -> Path:
        digest = hashlib.sha1(
            f"{self.client.model}|{chunk.id}|{chunk.text}".encode("utf-8")
        ).hexdigest()[:16]
        return self.cache_dir / f"{digest}.json"

    def _cached(self, chunk: Chunk) -> Optional[Dict[str, Any]]:
        target = self._cache_file(chunk)
        if not target.exists():
            return None
        try:
            return json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def _store(self, chunk: Chunk, payload: Dict[str, Any]) -> None:
        self._cache_file(chunk).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    # ---- 抽取 -----------------------------------------------------------

    def _render_prompt(self, chunk: Chunk, document: CleanedDocument) -> str:
        entity_lines = "\n".join(
            f"- {name}（{label}）" for name, label in self.schema.describe()["entity_types"].items()
        )
        relation_lines = "\n".join(
            f"- {name}({'/'.join(body['domain'])} -> {'/'.join(body['range'])})：{body['label']}"
            for name, body in self.schema.describe()["relation_types"].items()
        )
        return self._user_template.format(
            doc_title=document.title,
            doc_type=document.entity_type,
            entity_types=entity_lines,
            relation_types=relation_lines,
            chunk_text=chunk.text[: self.max_chars],
        )

    def extract(
        self, chunk: Chunk, document: CleanedDocument
    ) -> Tuple[List[Entity], List[Relation]]:
        payload = self._cached(chunk)
        if payload is None:
            raw = self.client.complete(self._system_prompt, self._render_prompt(chunk, document))
            try:
                payload = parse_json_block(raw)
            except ValueError as exc:
                raise LLMUnavailable(f"片段 {chunk.id} 的返回无法解析: {exc}") from exc
            if not isinstance(payload, dict):
                raise LLMUnavailable(f"片段 {chunk.id} 的返回结构不是对象")
            self._store(chunk, payload)
        return self._assemble(payload, chunk, document)

    def _assemble(
        self, payload: Dict[str, Any], chunk: Chunk, document: CleanedDocument
    ) -> Tuple[List[Entity], List[Relation]]:
        declared: Dict[str, str] = {}
        aliases: Dict[str, List[str]] = {}
        for item in payload.get("entities", []) or []:
            name = str(item.get("name", "")).strip()
            entity_type = self.schema.canonical_entity_type(str(item.get("type", "")))
            if not name or not entity_type:
                self.rejected.append(
                    {"kind": "entity", "chunk_id": chunk.id, "payload": item, "reason": "类型不在本体内"}
                )
                continue
            declared[name] = entity_type
            aliases[name] = [str(a).strip() for a in item.get("aliases", []) or [] if str(a).strip()]

        entities: Dict[str, Entity] = {}
        relations: Dict[str, Relation] = {}

        for item in payload.get("relations", []) or []:
            head = str(item.get("head", "")).strip()
            tail = str(item.get("tail", "")).strip()
            relation_type = self.schema.canonical_relation_type(str(item.get("relation", "")))
            head_type = self.schema.canonical_entity_type(
                str(item.get("head_type") or declared.get(head, ""))
            )
            tail_type = self.schema.canonical_entity_type(
                str(item.get("tail_type") or declared.get(tail, ""))
            )
            confidence = float(item.get("confidence", 0.7) or 0.7)

            if not head or not tail or not relation_type or not head_type or not tail_type:
                self.rejected.append(
                    {"kind": "relation", "chunk_id": chunk.id, "payload": item, "reason": "字段缺失或类型不可识别"}
                )
                continue
            if confidence < self.min_confidence:
                self.rejected.append(
                    {"kind": "relation", "chunk_id": chunk.id, "payload": item, "reason": "置信度低于阈值"}
                )
                continue
            reason = self.schema.validate_triple(head_type, relation_type, tail_type)
            if reason:
                self.rejected.append(
                    {"kind": "relation", "chunk_id": chunk.id, "payload": item, "reason": reason}
                )
                continue

            quote = str(item.get("quote", "")).strip()
            located = chunk.text.find(quote) if quote else -1
            if located >= 0:
                start = chunk.char_offset + located
                end = start + len(quote)
                snippet = quote
            else:
                start, end = chunk.char_offset, chunk.char_offset + len(chunk.text)
                snippet = chunk.text
            evidence = Evidence(
                doc_id=document.doc_id,
                chunk_id=chunk.id,
                char_start=start,
                char_end=end,
                raw_text=snippet,
                confidence=confidence,
            )

            for name, entity_type in ((head, head_type), (tail, tail_type)):
                key = entity_key(name, entity_type)
                entity = entities.get(key)
                if entity is None:
                    entity = Entity(
                        id=key, name=name, type=entity_type, aliases=list(aliases.get(name, []))
                    )
                    entities[key] = entity
                entity.evidences.append(evidence)

            head_key = entity_key(head, head_type)
            tail_key = entity_key(tail, tail_type)
            relation_id = f"{head_key}|{relation_type}|{tail_key}"
            relation = relations.get(relation_id)
            if relation is None:
                relation = Relation(
                    id=relation_id,
                    head_id=head_key,
                    tail_id=tail_key,
                    type=relation_type,
                    start_year=_year_or_none(item.get("start_year")),
                    end_year=_year_or_none(item.get("end_year")),
                    attributes={"extractor": self.name},
                )
                relations[relation_id] = relation
            relation.evidences.append(evidence)

        return list(entities.values()), list(relations.values())
