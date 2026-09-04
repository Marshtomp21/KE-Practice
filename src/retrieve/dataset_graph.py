"""Build the local retrieval graph from the course movie dataset.

The two research methods use this adapter instead of changing the vector and
Neo4j baselines.  Dataset relations are converted to the project's canonical
edge direction and linked back to persisted chunks as evidence.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from ..core.config import Settings
from ..core.types import Chunk, Entity, Evidence, Relation
from ..graph.networkx_store import NetworkxGraphStore
from .vector_index import ChunkVectorIndex


RELATION_MAPPING: Dict[str, Tuple[str, bool, str]] = {
    # dataset label: (schema relation, reverse source/target, target type)
    "执导": ("directed", True, "Person"),
    "出演": ("acted_in", True, "Person"),
    "编剧": ("wrote", True, "Person"),
    "出品": ("produced", True, "Company"),
    "获奖": ("won", False, "Award"),
    "提名": ("nominated", False, "Award"),
    "类型": ("has_genre", False, "Genre"),
    "改编自": ("adapted_from", False, "Movie"),
    "前作": ("sequel_of", False, "Movie"),
    # If A's record names B as its sequel, B is the sequel of A.
    "续作": ("sequel_of", True, "Movie"),
}

_NOISE_NAME = re.compile(r"[{}\[\]|<>]|^\s*$")


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _valid_entity(entity_id: object, name: object) -> bool:
    token_id = str(entity_id or "").strip()
    token_name = str(name or "").strip()
    return bool(token_id and token_name and len(token_name) <= 80 and not _NOISE_NAME.search(token_name))


def _relation_id(head_id: str, relation_type: str, tail_id: str) -> str:
    value = f"{head_id}|{relation_type}|{tail_id}"
    return "dataset-" + hashlib.sha1(value.encode("utf-8")).hexdigest()


class DatasetGraphLoader:
    """Convert tracked movie records into a provenance-bearing in-memory graph."""

    def __init__(self, settings: Settings, index: ChunkVectorIndex) -> None:
        self.settings = settings
        self.index = index
        self.dataset_dir = settings.path("paths.dataset_dir")
        self.chunks_by_doc: Dict[str, List[Chunk]] = {}
        for chunk in index.chunks:
            self.chunks_by_doc.setdefault(chunk.doc_id, []).append(chunk)
        self._entities: Dict[str, Entity] = {}

    def load(self) -> NetworkxGraphStore:
        required = ("films.jsonl", "actors.jsonl", "directors.jsonl", "relations.jsonl")
        missing = [name for name in required if not (self.dataset_dir / name).exists()]
        if missing:
            raise FileNotFoundError(f"电影数据集缺少文件 {missing}：{self.dataset_dir}")

        self._read_named_entities()
        raw_relations = list(_iter_jsonl(self.dataset_dir / "relations.jsonl"))
        for row in raw_relations:
            self._ensure_relation_entities(row)

        store = NetworkxGraphStore()
        for entity in self._entities.values():
            store.upsert_entity(entity)
        for row in raw_relations:
            relation = self._convert_relation(row)
            if relation is not None:
                store.upsert_relation(relation)
        return store

    def _read_named_entities(self) -> None:
        for row in _iter_jsonl(self.dataset_dir / "films.jsonl"):
            film = row.get("film") or {}
            self._add_entity(film.get("id"), film.get("name"), "Movie", film.get("aliases") or [])
        for file_name in ("actors.jsonl", "directors.jsonl"):
            for row in _iter_jsonl(self.dataset_dir / file_name):
                person = row.get("person") or {}
                self._add_entity(
                    person.get("id"), person.get("name"), "Person", person.get("aliases") or []
                )

    def _ensure_relation_entities(self, row: dict) -> None:
        label = str(row.get("relation") or "").strip()
        _, _, target_type = RELATION_MAPPING.get(label, (label or "RELATED_TO", False, "Entity"))
        source_type = "Movie" if row.get("source_type") == "影片" else "Entity"
        self._add_entity(row.get("source_id"), row.get("source_name"), source_type, [])
        self._add_entity(row.get("target_id"), row.get("target_name"), target_type, [])

    def _add_entity(
        self, entity_id: object, name: object, entity_type: str, aliases: Sequence[object]
    ) -> None:
        if not _valid_entity(entity_id, name):
            return
        key = str(entity_id).strip()
        display = str(name).strip()
        clean_aliases = []
        for alias in aliases:
            token = str(alias or "").strip()
            if token and token != display and not _NOISE_NAME.search(token) and token not in clean_aliases:
                clean_aliases.append(token)
        existing = self._entities.get(key)
        if existing is not None:
            for alias in [display, *clean_aliases]:
                if alias != existing.name and alias not in existing.aliases:
                    existing.aliases.append(alias)
            if existing.type == "Entity" and entity_type != "Entity":
                existing.type = entity_type
            return

        evidence = self._entity_evidence(key, entity_type, display)
        self._entities[key] = Entity(
            id=key,
            name=display,
            type=entity_type,
            aliases=clean_aliases,
            attributes={"source": "movie_dataset"},
            evidences=[evidence] if evidence else [],
        )

    def _entity_evidence(self, entity_id: str, entity_type: str, name: str) -> Optional[Evidence]:
        if entity_type == "Movie":
            candidates = self.chunks_by_doc.get(f"film_{entity_id}", [])
        else:
            candidates = [
                *self.chunks_by_doc.get(f"person_actor_{entity_id}", []),
                *self.chunks_by_doc.get(f"person_director_{entity_id}", []),
            ]
        return self._locate(candidates, name)

    def _convert_relation(self, row: dict) -> Optional[Relation]:
        source_id = str(row.get("source_id") or "").strip()
        target_id = str(row.get("target_id") or "").strip()
        if source_id not in self._entities or target_id not in self._entities:
            return None
        raw_type = str(row.get("relation") or "RELATED_TO").strip()
        relation_type, reverse, _ = RELATION_MAPPING.get(raw_type, (raw_type, False, "Entity"))
        head_id, tail_id = (target_id, source_id) if reverse else (source_id, target_id)
        target_name = str(row.get("target_name") or "").strip()
        candidates = self.chunks_by_doc.get(f"film_{source_id}", [])
        evidence = self._locate(candidates, target_name)
        attributes = {
            "source": "movie_dataset",
            "dataset_relation": raw_type,
            "evidence_url": str(row.get("evidence_url") or ""),
        }
        roles = list(row.get("roles") or [])
        if roles:
            attributes["roles"] = roles
        return Relation(
            id=_relation_id(head_id, relation_type, tail_id),
            head_id=head_id,
            tail_id=tail_id,
            type=relation_type,
            attributes=attributes,
            evidences=[evidence] if evidence else [],
        )

    @staticmethod
    def _locate(candidates: Sequence[Chunk], needle: str) -> Optional[Evidence]:
        if not candidates:
            return None
        selected = candidates[0]
        local_start = 0
        if needle:
            for chunk in candidates:
                position = chunk.text.find(needle)
                if position >= 0:
                    selected = chunk
                    local_start = position
                    break
        width = len(needle) if needle and needle in selected.text else len(selected.text)
        raw_text = selected.text[local_start : local_start + width]
        return Evidence(
            doc_id=selected.doc_id,
            chunk_id=selected.id,
            char_start=selected.char_offset + local_start,
            char_end=selected.char_offset + local_start + width,
            raw_text=raw_text,
            confidence=1.0,
        )
