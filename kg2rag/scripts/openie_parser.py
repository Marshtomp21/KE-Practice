"""Dependency-free parsing helpers for KG²RAG OpenIE responses."""
from __future__ import annotations

import json
import re
from typing import Any


XML_TRIPLE = re.compile(r"<([^<>]+)>")
FENCE = re.compile(r"```(?:json|xml)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
TRIPLE_BLOCK = re.compile(r"<trip(?:le|let)>\s*(.*?)\s*</trip(?:le|let)>", re.IGNORECASE | re.DOTALL)


def normalized(value: str) -> str:
    return re.sub(r"\W+", "", value).casefold()


def _clean_triple(values: list[Any]) -> list[str] | None:
    if len(values) != 3:
        return None
    triple = [str(value).strip() for value in values]
    return triple if all(triple) else None


def _triples_from_json(value: Any) -> list[list[str]]:
    if isinstance(value, dict):
        for key in ("triples", "relations", "relationships", "data"):
            if key in value:
                return _triples_from_json(value[key])
        fields = [
            ("subject", "relation", "object"), ("subject", "predicate", "object"),
            ("head", "relation", "tail"), ("head", "predicate", "tail"),
            ("source", "relation", "target"), ("source", "type", "target"),
        ]
        for field_names in fields:
            if all(name in value for name in field_names):
                triple = _clean_triple([value[name] for name in field_names])
                return [triple] if triple else []
        return []
    if isinstance(value, list):
        triple = _clean_triple(value)
        if triple and all(not isinstance(item, (dict, list)) for item in value):
            return [triple]
        rows: list[list[str]] = []
        for item in value:
            rows.extend(_triples_from_json(item))
        return rows
    return []


def parse_triples(response: str) -> list[list[str]]:
    """Accept JSON objects/lists and common XML OpenIE response variants."""
    body = response.strip()
    fenced = FENCE.search(body)
    if fenced:
        body = fenced.group(1).strip()
    parsed: list[list[str]] = []
    try:
        parsed = _triples_from_json(json.loads(body))
    except json.JSONDecodeError:
        pass
    if not parsed:
        for block in TRIPLE_BLOCK.findall(body):
            fields = []
            for name in ("subject", "relation", "predicate", "object"):
                match = re.search(fr"<{name}>\s*(.*?)\s*</{name}>", block, re.IGNORECASE | re.DOTALL)
                if match and (name != "predicate" or not any(x[0] == "relation" for x in fields)):
                    fields.append((name, match.group(1)))
            values = [
                next((value for field, value in fields if field == "subject"), ""),
                next((value for field, value in fields if field in ("relation", "predicate")), ""),
                next((value for field, value in fields if field == "object"), ""),
            ]
            triple = _clean_triple(values)
            if triple:
                parsed.append(triple)
            else:
                triple = _clean_triple(re.split(r"\s*(?:##|\|)\s*", re.sub(r"<[^>]+>", "", block)))
                if triple:
                    parsed.append(triple)
        for token in XML_TRIPLE.findall(body):
            triple = _clean_triple(re.split(r"\s*(?:##|\|)\s*", token))
            if triple:
                parsed.append(triple)
        for match in re.findall(r"<triple>\s*(.*?)\s*</triple>", body, re.IGNORECASE | re.DOTALL):
            triple = _clean_triple(re.split(r"\s*(?:##|\|)\s*", match))
            if triple:
                parsed.append(triple)
    unique: dict[tuple[str, str, str], list[str]] = {}
    for row in parsed:
        unique[tuple(normalized(item) for item in row)] = row
    return list(unique.values())
