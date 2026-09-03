"""Read-only quality audit for the shared movie corpus.

This script only reads the dataset path supplied on the command line and prints
a JSON report to stdout. It deliberately does not modify the team worktrees.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


REVIEW_SECTION_KEYWORDS = (
    "评价",
    "评论",
    "反响",
    "回响",
    "反应",
    "評價",
    "評論",
    "迴響",
    "影评",
    "评分",
    "争议",
    "批评",
    "荣誉",
    "奖项",
    "票房",
)
MARKUP_PATTERN = re.compile(r"\{\{|\}\}|\[\[|\]\]|-\{[^}]+\}-|<[^>]+>")
HEADING_PATTERN = re.compile(r"^\s*={2,}\s*([^=\n]+?)\s*={2,}\s*$", re.MULTILINE)


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return records


def quantiles(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "median": 0, "p90": 0, "max": 0}
    ordered = sorted(values)
    p90_index = min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))
    return {
        "min": ordered[0],
        "median": median(ordered),
        "p90": ordered[p90_index],
        "max": ordered[-1],
    }


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.size: dict[str, int] = {}

    def find(self, item: str) -> str:
        if item not in self.parent:
            self.parent[item] = item
            self.size[item] = 1
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]

    def component_sizes(self) -> list[int]:
        counts: Counter[str] = Counter(self.find(item) for item in self.parent)
        return sorted(counts.values(), reverse=True)


def audit(base: Path) -> dict:
    films = read_jsonl(base / "films.jsonl")
    actors = read_jsonl(base / "actors.jsonl")
    directors = read_jsonl(base / "directors.jsonl")
    relations = read_jsonl(base / "relations.jsonl")
    collaborations = read_jsonl(base / "collaborations.jsonl")

    film_ids = [record.get("film", {}).get("id") for record in films]
    film_names = [record.get("film", {}).get("name") for record in films]
    raw_lengths = []
    intro_lengths = []
    review_section_counts: Counter[str] = Counter()
    films_with_review_sections = 0
    films_with_any_review_terms = 0
    malformed_film_names = []

    for record in films:
        film = record.get("film") or {}
        name = str(film.get("name") or "")
        source = record.get("source_document") or {}
        raw = str(source.get("raw_wikitext") or "")
        intro = str(source.get("intro") or "")
        raw_lengths.append(len(raw))
        intro_lengths.append(len(intro))
        headings = HEADING_PATTERN.findall(raw)
        matched = [
            heading.strip()
            for heading in headings
            if any(keyword in heading for keyword in REVIEW_SECTION_KEYWORDS)
        ]
        if matched:
            films_with_review_sections += 1
            review_section_counts.update(matched)
        if any(keyword in raw for keyword in REVIEW_SECTION_KEYWORDS):
            films_with_any_review_terms += 1
        if MARKUP_PATTERN.search(name):
            malformed_film_names.append(name)

    relation_types = Counter(str(row.get("relation") or "") for row in relations)
    relations_with_evidence_url = sum(bool(row.get("evidence_url")) for row in relations)
    relations_with_raw_evidence = sum(bool(row.get("raw_evidence")) for row in relations)
    temporary_target_ids = sum(
        str(row.get("target_id") or "").startswith("NAME:") for row in relations
    )

    malformed_entity_names = Counter()
    malformed_examples: dict[str, list[str]] = defaultdict(list)
    for entity_type, records, key in (
        ("演员", actors, "person"),
        ("导演", directors, "person"),
    ):
        for record in records:
            name = str((record.get(key) or {}).get("name") or "")
            if MARKUP_PATTERN.search(name):
                malformed_entity_names[entity_type] += 1
                if len(malformed_examples[entity_type]) < 5:
                    malformed_examples[entity_type].append(name)

    actor_text_coverage = sum(bool(row.get("has_wikipedia_text")) for row in actors)
    director_text_coverage = sum(bool(row.get("has_wikipedia_text")) for row in directors)

    graph = UnionFind()
    for row in relations:
        source = str(row.get("source_id") or "")
        target = str(row.get("target_id") or "")
        if source and target:
            graph.union(source, target)
    component_sizes = graph.component_sizes()

    return {
        "dataset_path": str(base.resolve()),
        "record_counts": {
            "films": len(films),
            "actors": len(actors),
            "directors": len(directors),
            "relations": len(relations),
            "collaborations": len(collaborations),
        },
        "film_quality": {
            "duplicate_ids": len(film_ids) - len(set(film_ids)),
            "duplicate_names": len(film_names) - len(set(film_names)),
            "films_with_intro": sum(length > 0 for length in intro_lengths),
            "films_with_raw_wikitext": sum(length > 0 for length in raw_lengths),
            "intro_character_lengths": quantiles(intro_lengths),
            "raw_wikitext_character_lengths": quantiles(raw_lengths),
            "films_with_review_like_sections": films_with_review_sections,
            "review_like_section_coverage": round(films_with_review_sections / len(films), 4),
            "films_with_review_terms_anywhere": films_with_any_review_terms,
            "common_review_like_headings": review_section_counts.most_common(20),
            "malformed_name_count": len(malformed_film_names),
            "malformed_name_examples": malformed_film_names[:5],
        },
        "person_quality": {
            "actors_with_text": actor_text_coverage,
            "actor_text_coverage": round(actor_text_coverage / len(actors), 4),
            "directors_with_text": director_text_coverage,
            "director_text_coverage": round(director_text_coverage / len(directors), 4),
            "malformed_name_counts": dict(malformed_entity_names),
            "malformed_name_examples": dict(malformed_examples),
        },
        "relation_quality": {
            "relation_type_counts": dict(relation_types),
            "with_evidence_url": relations_with_evidence_url,
            "with_raw_evidence": relations_with_raw_evidence,
            "temporary_target_ids": temporary_target_ids,
            "temporary_target_id_rate": round(temporary_target_ids / len(relations), 4),
        },
        "graph_connectivity_without_derived_collaborations": {
            "nodes": len(graph.parent),
            "components": len(component_sizes),
            "largest_component_nodes": component_sizes[0] if component_sizes else 0,
            "largest_component_ratio": round(
                component_sizes[0] / len(graph.parent), 4
            ) if component_sizes else 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    report = audit(args.dataset)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
