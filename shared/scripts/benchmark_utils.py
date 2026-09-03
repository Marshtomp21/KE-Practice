#!/usr/bin/env python3
"""Shared, auditable scoring helpers for the reproduction benchmark."""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable


ABSTENTION_MARKERS = (
    "无法回答", "无法判断", "无法确定", "无法根据", "无法从", "不能回答", "不能确定", "不足以回答",
    "信息不足", "资料不足", "缺少足够", "没有足够", "未提供", "没有信息表明",
    "无相关信息", "不知道", "不清楚", "does not contain information", "doesn't contain information",
    "not enough information", "cannot determine", "cannot be determined", "not provided",
    "unable to answer", "unable to determine",
)
REFERENCE_HEADINGS = ("### references", "## references", "references:", "参考资料", "引用：")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return "".join(ch for ch in value if ch.isalnum())


def strip_references(answer: str) -> str:
    lowered = answer.casefold()
    positions = [lowered.find(marker) for marker in REFERENCE_HEADINGS if lowered.find(marker) >= 0]
    return answer[: min(positions)] if positions else answer


def begins_with_abstention(answer: str) -> bool:
    body = strip_references(answer).strip()
    first = body.split("\n\n", 1)[0][:220].casefold()
    return any(marker in first for marker in ABSTENTION_MARKERS)


def entity_aliases(entity: dict[str, Any]) -> list[str]:
    values = [entity.get("name", ""), *(entity.get("aliases") or [])]
    seen: set[str] = set()
    aliases: list[str] = []
    for value in values:
        token = normalize_text(str(value))
        if token and token not in seen:
            seen.add(token)
            aliases.append(token)
    return aliases


def mentioned_entities(answer: str, entities: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = normalize_text(strip_references(answer))
    return [entity for entity in entities if any(alias in normalized for alias in entity_aliases(entity))]


def project_entity_set_answer(answer: str) -> tuple[str, str]:
    """Select the explicit prediction portion of a free-form list answer.

    Explanatory RAG answers often mention every candidate while comparing film
    casts.  Counting those evidence mentions as predictions is incorrect.  The
    deterministic precedence below is intentionally small and auditable:
    a labelled conclusion wins; otherwise use the first contiguous Markdown
    bullet block; otherwise score the whole answer body.
    """
    body = strip_references(answer).strip()
    conclusion = re.search(
        r"(?:^|\n)\s*(?:#{1,6}\s*)?(?:\*\*)?(?:结论|最终答案|总结)(?:\*\*)?\s*(?:[：:]\s*|\n+)",
        body,
    )
    if conclusion:
        return body[conclusion.end():], "labelled conclusion"
    summary = re.search(r"(?:以上(?:信息)?(?:显示|表明)|综上(?:所述)?)[，,:：]?\s*", body)
    if summary:
        return body[summary.end():], "explicit summary sentence"
    lines = body.splitlines()
    bullet_indexes = [index for index, line in enumerate(lines) if re.match(r"^\s*[-*]\s+", line)]
    if bullet_indexes:
        selected: list[str] = []
        index = bullet_indexes[0]
        while index < len(lines):
            line = lines[index]
            if re.match(r"^\s*[-*]\s+", line):
                selected.append(line)
            elif selected and line.strip():
                break
            index += 1
        return "\n".join(selected), "first markdown bullet block"
    return body, "whole answer"


def _ids(entities: Iterable[dict[str, Any]]) -> set[str]:
    return {str(item["canonical_id"]) for item in entities}


def score_answer(question: dict[str, Any], answer: str) -> dict[str, Any]:
    """Score one answer and return a fully inspectable metric record."""
    answer_kind = question.get("answer_kind", "entity")
    answerable = bool(question.get("answerable", True))
    abstained = begins_with_abstention(answer)
    if answer_kind == "abstention" or not answerable:
        score = 1.0 if abstained else 0.0
        return {
            "metric": "abstention_accuracy", "score": score, "precision": score,
            "recall": score, "f1": score, "exact_match": bool(score),
            "abstained": abstained, "matched_gold_ids": [], "predicted_entity_ids": [],
            "reason": "explicit abstention detected" if abstained else "no explicit abstention detected",
        }

    gold = list(question.get("gold_answers") or [])
    candidates = list(question.get("answer_candidates") or gold)
    prediction_text, projection = (
        project_entity_set_answer(answer) if answer_kind == "entity_set"
        else (strip_references(answer), "whole answer")
    )
    predicted = mentioned_entities(prediction_text, candidates)
    gold_ids = _ids(gold)
    predicted_ids = _ids(predicted)
    matched = gold_ids & predicted_ids
    if abstained:
        precision = recall = f1 = 0.0
        reason = "answer begins with an abstention"
    else:
        precision = len(matched) / len(predicted_ids) if predicted_ids else 0.0
        recall = len(matched) / len(gold_ids) if gold_ids else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        reason = f"canonical entity alias matching; prediction projection: {projection}"

    if answer_kind == "entity":
        score = f1
        exact = predicted_ids == gold_ids and bool(gold_ids) and not abstained
        metric = "entity_em_f1"
    elif answer_kind == "entity_set":
        score = f1
        exact = predicted_ids == gold_ids and bool(gold_ids) and not abstained
        metric = "entity_set_f1"
    else:
        raise ValueError(f"Unsupported answer_kind: {answer_kind!r}")
    return {
        "metric": metric, "score": round(score, 6), "precision": round(precision, 6),
        "recall": round(recall, 6), "f1": round(f1, 6), "exact_match": exact,
        "abstained": abstained, "matched_gold_ids": sorted(matched),
        "predicted_entity_ids": sorted(predicted_ids), "reason": reason,
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"count": 0, "mean_score": 0.0, "exact_match_rate": 0.0, "by_type": {}}
    by_type: dict[str, list[float]] = {}
    for row in results:
        by_type.setdefault(str(row.get("type", "unknown")), []).append(float(row["score"]))
    return {
        "count": len(results),
        "mean_score": round(sum(float(row["score"]) for row in results) / len(results), 4),
        "exact_match_rate": round(sum(bool(row.get("metrics", {}).get("exact_match")) for row in results) / len(results), 4),
        "by_type": {key: {"count": len(values), "mean_score": round(sum(values) / len(values), 4)} for key, values in sorted(by_type.items())},
    }


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "to_dict"):
        return json_safe(value.to_dict())
    return str(value)
