"""Benchmark v2 的严格答案、文档和关系路径评分。"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import yaml

from ..core.types import Answer

CITATION = re.compile(r"\[(?:S|G)\d+\]", re.I)
MOVIE_QUOTE = re.compile(r"《([^》]{1,80})》")
RAW_EVIDENCE_LINE = re.compile(r"^\s*(?:[-*]\s*)?\[S\d+\]\s*原文[：:]", re.I)
SECTION_CUTOFF = re.compile(r"^\s*(?:#{1,6}\s*)?(?:依据|证据|原文片段|参考材料)(?:\s*[：:].*)?\s*$")
RELATION_ALIASES = {
    "执导": "directed", "directed": "directed",
    "出演": "acted_in", "acted_in": "acted_in",
}


def normalize(text: object) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return re.sub(r"[\s\W_]+", "", value)


def load_benchmark(path: Path | str) -> dict:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    questions = payload.get("questions") or []
    declared = int((payload.get("benchmark") or {}).get("question_count") or 0)
    if not questions or declared != len(questions):
        raise ValueError(f"benchmark 题量声明与内容不一致：{declared} / {len(questions)}")
    return payload


def answer_body(text: str) -> str:
    """剔除原文转录和证据附录，避免 gold 只出现在引用里也被算作回答正确。"""
    kept: List[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if RAW_EVIDENCE_LINE.match(stripped):
            continue
        if SECTION_CUTOFF.match(stripped):
            break
        kept.append(line)
    return CITATION.sub("", "\n".join(kept)).strip()


def prf(predicted: set[str], gold: set[str]) -> Tuple[float, float, float, float]:
    if not gold:
        exact = 1.0 if not predicted else 0.0
        return exact, exact, exact, exact
    true_positive = len(predicted & gold)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, float(predicted == gold)


class BenchmarkScorer:
    def __init__(self, benchmark: Mapping[str, Any]) -> None:
        self.benchmark = benchmark
        self.catalog: Dict[str, dict] = dict(benchmark.get("entity_catalog") or {})
        self.surface_index: Dict[str, Dict[str, set[str]]] = {
            "Movie": {}, "Person": {},
        }
        for entity_id, item in self.catalog.items():
            entity_type = str(item.get("type") or "")
            if entity_type not in self.surface_index:
                continue
            for surface in [item.get("name"), *(item.get("aliases") or [])]:
                token = normalize(surface)
                if token:
                    self.surface_index[entity_type].setdefault(token, set()).add(entity_id)

    def score(
        self, question: Mapping[str, Any], answer: Answer, graph_view: str = "masked",
    ) -> Dict[str, Any]:
        body = answer_body(answer.text)
        if question.get("answer_type") == "no_answer":
            answer_metrics = self._score_no_answer(body)
            predicted: set[str] = set()
        else:
            target_type = str(question.get("target_entity_type") or "")
            excluded = {item["id"] for item in question.get("subjects") or []}
            gold = {item["id"] for item in question.get("gold_answers") or []}
            predicted = self.extract_entities(body, target_type, gold | excluded) - excluded
            precision, recall, f1, exact = prf(predicted, gold)
            answer_metrics = {
                "precision": precision, "recall": recall, "f1": f1, "exact_match": exact,
                "predicted_ids": sorted(predicted), "gold_ids": sorted(gold),
            }

        citations = {citation.doc_id for citation in answer.citations}
        gold_documents = set(question.get("gold_documents") or [])
        doc_precision, doc_recall, doc_f1, doc_exact = prf(citations, gold_documents)
        perturbation = dict(question.get("graph_perturbation") or {})
        nested_debug = answer.debug_info.get("retrieval")
        retrieval_debug = dict(nested_debug if isinstance(nested_debug, Mapping) else answer.debug_info)
        expected_gap = bool(perturbation.get("expected_gap")) and graph_view == "masked"
        compensation_gold = (
            set(perturbation.get("compensation_gold_documents") or []) if expected_gap else set()
        )
        compensation_docs = set(retrieval_debug.get("compensation_documents") or [])
        comp_precision, comp_recall, comp_f1, _ = prf(compensation_docs, compensation_gold)
        gap_detected = bool(retrieval_debug.get("gap_detected", False))
        compensation_triggered = bool(retrieval_debug.get("compensation_triggered", False))
        edge_support: Dict[Tuple[str, str, str], set[str]] = {}
        for evidence in perturbation.get("compensation_gold_evidence") or []:
            edge = evidence.get("supports") or {}
            key = self._directed_edge_key(edge)
            if all(key):
                edge_support.setdefault(key, set()).add(str(evidence.get("doc_id") or ""))
        temporary_relations = self._supported_temporary_relations(retrieval_debug, edge_support)
        relation_metrics = self._score_relations(question, answer, temporary_relations)
        gold_names = [str(item.get("name") or "") for item in question.get("gold_answers") or []]
        legacy_recall = (
            sum(name in answer.text for name in gold_names) / len(gold_names) if gold_names else 0.0
        )
        return {
            "answer": answer_metrics,
            "retrieval": {
                "document_precision": doc_precision,
                "document_recall": doc_recall,
                "document_f1": doc_f1,
                "document_exact": doc_exact,
                "retrieved_documents": sorted(citations),
                "gold_documents": sorted(gold_documents),
                **relation_metrics,
            },
            "gap": {
                "expected_gap": expected_gap,
                "gap_detected": gap_detected,
                "gap_detection_correct": float(gap_detected == expected_gap),
                "compensation_triggered": compensation_triggered,
                "unnecessary_compensation": float(compensation_triggered and not expected_gap),
                "compensation_document_precision": comp_precision if expected_gap else None,
                "compensation_document_recall": comp_recall if expected_gap else None,
                "compensation_document_f1": comp_f1 if expected_gap else None,
                "compensation_documents": sorted(compensation_docs),
                "compensation_gold_documents": sorted(compensation_gold),
            },
            "diagnostics": {
                "scored_answer_body": body,
                "legacy_gold_substring_recall": legacy_recall,
            },
        }

    def extract_entities(
        self, text: str, entity_type: str, always_allow: Iterable[str] = (),
    ) -> set[str]:
        normalized_text = normalize(text)
        allowed_short = set(always_allow)
        predicted: set[str] = set()
        # 片名优先读取书名号；同时保留全文扫描，以兼容不加书名号的简短答案。
        movie_tokens = {normalize(item) for item in MOVIE_QUOTE.findall(text)}
        covered: List[Tuple[int, int]] = []
        surfaces = sorted(self.surface_index.get(entity_type, {}).items(), key=lambda item: -len(item[0]))
        for surface, ids in surfaces:
            if len(surface) < 2:
                continue
            if len(surface) < 3 and not (ids & allowed_short):
                continue
            spans = [match.span() for match in re.finditer(re.escape(surface), normalized_text)]
            if entity_type == "Movie" and movie_tokens:
                if surface in movie_tokens and not spans:
                    spans = [(-1, -1)]
            spans = [span for span in spans
                     if span == (-1, -1) or not any(span[0] >= lo and span[1] <= hi for lo, hi in covered)]
            if spans:
                predicted.update(ids)
                covered.extend(span for span in spans if span != (-1, -1))
        return predicted

    @staticmethod
    def _score_no_answer(body: str) -> Dict[str, Any]:
        compact = normalize(body)
        specific_denial = bool(
            re.search(r"(?:无|没有|未发现|不存在).{0,12}(?:共同|同一部|交集)", body)
            or re.search(r"结论\s*[：:]\s*无(?:\W|$)", body)
            or compact in {"无", "否"}
        )
        affirmative = bool(re.search(r"(?:有|存在).{0,8}(?:共同出演|共同参演|同一部影片)", body))
        insufficient = bool(re.search(r"(?:材料|信息|证据).{0,8}(?:不足|不够)|无法(?:判断|确定|作答)", body))
        correct = float(specific_denial and not affirmative and not insufficient)
        return {
            "precision": correct, "recall": correct, "f1": correct, "exact_match": correct,
            "predicted_ids": [], "gold_ids": [], "denial_detected": specific_denial,
            "affirmative_detected": affirmative, "insufficient_detected": insufficient,
        }

    def _canonical_endpoint(self, raw_id: str, names: Mapping[str, str]) -> set[str]:
        if raw_id in self.catalog:
            return {raw_id}
        token = normalize(names.get(raw_id, raw_id))
        matches: set[str] = set()
        for index in self.surface_index.values():
            matches.update(index.get(token, set()))
        return matches

    @staticmethod
    def _directed_edge_key(relation: Mapping[str, Any]) -> Tuple[str, str, str]:
        relation_type = str(relation.get("relation") or relation.get("type") or "")
        return (
            str(relation.get("head_id") or ""),
            RELATION_ALIASES.get(relation_type, relation_type),
            str(relation.get("tail_id") or ""),
        )

    @staticmethod
    def _supported_temporary_relations(
        debug: Mapping[str, Any], edge_support: Mapping[Tuple[str, str, str], set[str]],
    ) -> List[Mapping[str, Any]]:
        """只接受确有 gold 文档支撑的临时边，防止方法凭空声明路径已修复。"""
        accepted: List[Mapping[str, Any]] = []
        for relation in debug.get("temporary_relations") or []:
            supporting = set(relation.get("supporting_documents") or [])
            gold_documents = edge_support.get(BenchmarkScorer._directed_edge_key(relation), set())
            if gold_documents and supporting & gold_documents:
                accepted.append(relation)
        return accepted

    def _score_relations(
        self,
        question: Mapping[str, Any],
        answer: Answer,
        temporary_relations: Sequence[Mapping[str, Any]] = (),
    ) -> Dict[str, Any]:
        names = {entity.id: entity.name for entity in answer.subgraph.entities}
        retrieved: set[Tuple[frozenset[str], str]] = set()
        for relation in answer.subgraph.relations:
            relation_type = RELATION_ALIASES.get(relation.type, relation.type)
            heads = self._canonical_endpoint(relation.head_id, names)
            tails = self._canonical_endpoint(relation.tail_id, names)
            for head in heads:
                for tail in tails:
                    retrieved.add((frozenset((head, tail)), relation_type))

        recovered = set(retrieved)
        for relation in temporary_relations:
            relation_type = RELATION_ALIASES.get(
                str(relation.get("relation") or relation.get("type") or ""),
                str(relation.get("relation") or relation.get("type") or ""),
            )
            head_id = str(relation.get("head_id") or "")
            tail_id = str(relation.get("tail_id") or "")
            if head_id and tail_id and relation_type:
                recovered.add((frozenset((head_id, tail_id)), relation_type))

        paths = list(question.get("gold_paths") or [])
        gold_edges = [edge for path in paths for edge in path.get("edges") or []]
        edge_hits = []
        for edge in gold_edges:
            key = (frozenset((edge["head_id"], edge["tail_id"])),
                   RELATION_ALIASES.get(edge["relation"], edge["relation"]))
            edge_hits.append(key in retrieved)
        path_hits = []
        recovered_path_hits = []
        for path in paths:
            required = [
                (frozenset((edge["head_id"], edge["tail_id"])),
                 RELATION_ALIASES.get(edge["relation"], edge["relation"]))
                for edge in path.get("edges") or []
            ]
            path_hits.append(bool(required) and all(edge in retrieved for edge in required))
            recovered_path_hits.append(bool(required) and all(edge in recovered for edge in required))
        return {
            "relation_recall": sum(edge_hits) / len(edge_hits) if edge_hits else None,
            "path_complete_rate": sum(path_hits) / len(path_hits) if path_hits else None,
            "recovered_path_complete_rate": (
                sum(recovered_path_hits) / len(recovered_path_hits)
                if recovered_path_hits else None
            ),
            "gold_relation_count": len(edge_hits),
            "retrieved_relation_count": len(answer.subgraph.relations),
            "temporary_relation_count": len(temporary_relations),
        }
