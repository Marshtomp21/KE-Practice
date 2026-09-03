#!/usr/bin/env python3
"""Build and freeze the audited L2 film benchmark (v2).

V2 deduplicates each film's cast and only treats an actor as repeated when the
same canonical actor occurs in at least two distinct films by the director.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from benchmark_utils import normalize_text


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT.parent.parent / "KE-Practice" / "data" / "source" / "wikipedia_300_films_final"
DEFAULT_OUT = ROOT / "benchmarks" / "l2_film_120_v2"
VERSION = "l2_film_120_v2"

# Manually reviewed identity repairs for conflicting IDs/spellings in the
# upstream extraction.  These are deliberately small and auditable instead of
# applying fuzzy matching to every same-named person.
CURATED_PEOPLE = {
    normalize_text(alias): entity
    for entity, aliases in [
        (
            {"canonical_id": "Q717432", "type": "Person", "name": "张震", "aliases": ["张震", "张震 (演员)"]},
            ["张震", "张震 (演员)"],
        ),
        (
            {"canonical_id": "Q35332", "type": "Person", "name": "毕·彼特", "aliases": ["毕·彼特", "布拉德·皮特", "布莱德·彼特"]},
            ["毕·彼特", "布拉德·皮特", "布莱德·彼特"],
        ),
        (
            {"canonical_id": "Q313705", "type": "Person", "name": "詹森·舒瓦兹曼", "aliases": ["詹森·舒瓦兹曼", "杰森·薛兹曼", "积逊·舒华沙曼"]},
            ["詹森·舒瓦兹曼", "杰森·薛兹曼", "积逊·舒华沙曼"],
        ),
    ]
    for alias in aliases
}

FACT_SPECS = {
    "test": [
        ("test-fact-01", "Q124291916"), ("test-fact-02", "Q108628759"),
        ("test-fact-03", "Q131690947"), ("test-fact-04", "Q124472773"),
        ("test-fact-05", "Q126209959"), ("test-fact-06", "Q108003049"),
        ("test-fact-07", "Q109345157"), ("test-fact-08", "Q113671585"),
    ],
    "dev": [("dev-fact-01", "Q123185887"), ("dev-fact-02", "Q155653")],
}
PATH_SPECS = {
    "test": [
        ("test-path-01", "Q109345157", "王俊凯", "苗苗 (演员)"),
        ("test-path-02", "Q123928072", "佐伊·索尔达娜", "卡拉·索菲亚·贾斯康"),
        ("test-path-03", "Q130288104", "吴君如", "张天赋"),
        ("test-path-04", "Q131311712", "游学修", "钟雪莹"),
        ("test-path-05", "Q129333360", "许贤", "滕毅康"),
    ],
    "dev": [("dev-path-01", "Q108003049", "黄子华", "邓丽欣")],
}
AGGREGATE_SPECS = {
    "test": [
        ("test-aggregate-01", "Q15919819"), ("test-aggregate-02", "Q55400"),
        ("test-aggregate-03", "Q354554"), ("test-aggregate-04", "Q184903"),
        ("test-aggregate-05", "Q55431"),
    ],
    "dev": [("dev-aggregate-01", "Q223687")],
}
NO_ANSWER_SPECS = {
    "test": [
        ("test-no-answer-01", "《知识工程测试片：不存在的回声》由谁执导？"),
        ("test-no-answer-02", "林海舟与周云岚通过哪部影片产生关联？"),
    ],
    "dev": [("dev-no-answer-01", "《知识工程测试片：未上映的城市》有哪些主演？")],
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def clean_base_name(name: str) -> str:
    return re.sub(r"\s*[（(](?:演员|导演|歌手|电影摄影师|艺人)[^）)]*[）)]\s*$", "", name).strip()


def valid_entity_name(name: str) -> bool:
    return bool(name and re.search(r"[\w\u3400-\u9fff]", name) and not re.search(r"[{}\[\]|]", name))


def identity_key(item: dict[str, Any]) -> str:
    name = str(item.get("name", "")).strip()
    curated = CURATED_PEOPLE.get(normalize_text(name))
    if curated:
        return curated["canonical_id"]
    return normalize_text(clean_base_name(name))


def unique_people(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate one film's cast/crew, preferring stable external IDs."""
    chosen: dict[str, dict[str, Any]] = {}
    for raw in items:
        name = str(raw.get("name", "")).strip()
        if not valid_entity_name(name):
            continue
        key = identity_key(raw)
        item = dict(raw)
        previous = chosen.get(key)
        if previous is None or (str(previous.get("id", "")).startswith("NAME:") and not str(item.get("id", "")).startswith("NAME:")):
            chosen[key] = item
    return list(chosen.values())


def names(record: dict[str, Any], field: str) -> list[str]:
    return [str(item["name"]) for item in unique_people(record.get(field, []))]


def record_id(record: dict[str, Any]) -> str:
    return str(record["film"]["id"])


def title(record: dict[str, Any]) -> str:
    return str(record["film"]["name"]).strip()


def derived_aliases(name: str, extra: Iterable[str] = ()) -> list[str]:
    values = [name, *(value for value in extra if value)]
    stripped = re.sub(r"\s*[（(][^）)]*[）)]\s*$", "", name).strip()
    if stripped and stripped != name:
        values.append(stripped)
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        value = str(value).strip()
        key = normalize_text(value)
        if valid_entity_name(value) and key and key not in seen:
            seen.add(key)
            output.append(value)
    return output


def load_people_catalog(source: Path) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    for filename in ("actors.jsonl", "directors.jsonl"):
        for row in read_jsonl(source / filename):
            person = row.get("person") or {}
            name = str(person.get("name", "")).strip()
            if not valid_entity_name(name):
                continue
            candidate = {
                "canonical_id": str(person.get("id") or f"NAME:{name}"), "type": "Person", "name": name,
                "aliases": derived_aliases(name, extra=person.get("aliases") or []),
            }
            key = normalize_text(name)
            current = catalog.get(key)
            if current is None or (str(current["canonical_id"]).startswith("NAME:") and not candidate["canonical_id"].startswith("NAME:")):
                catalog[key] = candidate
    return catalog


def person_entity(item: dict[str, Any], catalog: dict[str, dict[str, Any]]) -> dict[str, Any]:
    name = str(item["name"]).strip()
    curated = CURATED_PEOPLE.get(normalize_text(name))
    if curated:
        return dict(curated)
    base = catalog.get(normalize_text(name))
    if base:
        entity = dict(base)
        entity["aliases"] = derived_aliases(name, [*(base.get("aliases") or []), item.get("wiki_title", "")])
        return entity
    return {
        "canonical_id": str(item.get("id") or f"NAME:{name}"), "type": "Person", "name": name,
        "aliases": derived_aliases(name, [item.get("wiki_title", "")]),
    }


def movie_entity(record: dict[str, Any]) -> dict[str, Any]:
    return {"canonical_id": record_id(record), "type": "Movie", "name": title(record), "aliases": derived_aliases(title(record))}


def compact_document(record: dict[str, Any]) -> str:
    film = record["film"]
    intro = str(record.get("source_document", {}).get("intro", "")).strip()
    lines = [f"影片：{film['name']}", f"数据集影片ID：{film['id']}"]
    mapping = {
        "directors": "导演", "actors": "演员", "screenwriters": "编剧",
        "production_companies": "制片公司", "genres": "类型", "awards_received": "奖项",
        "nominations": "提名", "adapted_from": "改编自", "previous_works": "前作", "sequels": "续作",
    }
    for field, label in mapping.items():
        values = names(record, field)
        if values:
            lines.append(f"{label}：{'、'.join(values)}。")
    if intro:
        lines.extend(["简介：", intro])
    return "\n".join(lines)


def line_evidence(doc_id: str, text: str, prefix: str, supports: list[str]) -> dict[str, Any]:
    offset = 0
    for line in text.splitlines(keepends=True):
        clean = line.rstrip("\r\n")
        if clean.startswith(prefix):
            return {"doc_id": doc_id, "char_start": offset, "char_end": offset + len(clean), "quote": clean, "supports": supports}
        offset += len(line)
    raise ValueError(f"Missing evidence line {prefix!r} in {doc_id}")


def edge(source: dict[str, Any], relation: str, target: dict[str, Any]) -> dict[str, str]:
    return {"source_id": source["canonical_id"], "relation": relation, "target_id": target["canonical_id"]}


def find_person(record: dict[str, Any], field: str, wanted: str, catalog: dict[str, dict[str, Any]]) -> dict[str, Any]:
    key = normalize_text(wanted)
    for item in unique_people(record.get(field, [])):
        entity = person_entity(item, catalog)
        if key in {normalize_text(alias) for alias in entity["aliases"]}:
            return entity
    raise ValueError(f"{wanted!r} is not a deduplicated {field} member of {title(record)!r}")


def make_fact(split: str, qid: str, record: dict[str, Any], documents: dict[str, str], catalog: dict[str, dict[str, Any]], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    directors = [person_entity(item, catalog) for item in unique_people(record.get("directors", []))]
    if not directors:
        raise ValueError(f"No director for {title(record)}")
    movie = movie_entity(record)
    doc_id = f"film_{record_id(record)}"
    return {
        "schema_version": 2, "id": qid, "split": split, "type": "fact", "answer_kind": "entity",
        "question": f"《{title(record)}》是由谁执导的？", "reference_answer": "；".join(x["name"] for x in directors),
        "gold_answers": directors, "answer_candidates": candidates, "answerable": True, "gold_documents": [doc_id],
        "gold_evidence": [line_evidence(doc_id, documents[doc_id], "导演：", [x["canonical_id"] for x in directors])],
        "required_relation_paths": [{"nodes": [x["canonical_id"], movie["canonical_id"]], "edges": [edge(x, "directed", movie)]} for x in directors],
        "notes": "人工选定影片；导演实体与证据跨度由冻结源记录生成并复核。",
    }


def make_path(split: str, qid: str, record: dict[str, Any], left_name: str, right_name: str, documents: dict[str, str], catalog: dict[str, dict[str, Any]], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    left = find_person(record, "actors", left_name, catalog)
    right = find_person(record, "actors", right_name, catalog)
    movie = movie_entity(record)
    doc_id = f"film_{record_id(record)}"
    return {
        "schema_version": 2, "id": qid, "split": split, "type": "multi_hop", "answer_kind": "entity",
        "question": f"{left['name']}与{right['name']}通过哪部影片产生关联？", "reference_answer": movie["name"],
        "gold_answers": [movie], "answer_candidates": candidates, "answerable": True, "gold_documents": [doc_id],
        "gold_evidence": [line_evidence(doc_id, documents[doc_id], "演员：", [left["canonical_id"], right["canonical_id"], movie["canonical_id"]])],
        "required_relation_paths": [{"nodes": [left["canonical_id"], movie["canonical_id"], right["canonical_id"]], "edges": [edge(left, "acted_in", movie), edge(right, "acted_in", movie)]}],
        "notes": "两名演员在同一冻结影片记录中出现；路径端点和证据跨度已复核。",
    }


def aggregate_for_director(records: list[dict[str, Any]], director_id: str, catalog: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]]:
    films: list[dict[str, Any]] = []
    director = None
    appearances: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for record in records:
        directors = [person_entity(item, catalog) for item in unique_people(record.get("directors", []))]
        matched = next((item for item in directors if item["canonical_id"] == director_id), None)
        if not matched:
            continue
        director = matched
        films.append(record)
        for raw_actor in unique_people(record.get("actors", [])):
            actor = person_entity(raw_actor, catalog)
            appearances[actor["canonical_id"]].append((record, actor))
    if director is None or len(films) < 2:
        raise ValueError(f"Director {director_id} has fewer than two source films")
    repeated = {key: value for key, value in appearances.items() if len({record_id(r) for r, _ in value}) >= 2}
    if not repeated:
        raise ValueError(f"Director {director['name']} has no actor repeated across distinct films")
    return director, films, repeated


def make_aggregate(split: str, qid: str, records: list[dict[str, Any]], director_id: str, documents: dict[str, str], catalog: dict[str, dict[str, Any]]) -> dict[str, Any]:
    director, films, repeated = aggregate_for_director(records, director_id, catalog)
    gold = sorted((rows[0][1] for rows in repeated.values()), key=lambda item: item["name"])
    candidates_by_id: dict[str, dict[str, Any]] = {}
    for record in films:
        for raw in unique_people(record.get("actors", [])):
            actor = person_entity(raw, catalog)
            candidates_by_id[actor["canonical_id"]] = actor
    evidence: list[dict[str, Any]] = []
    paths = []
    for actor_id, rows in sorted(repeated.items()):
        actor = rows[0][1]
        actor_paths = []
        for record, _ in rows:
            movie = movie_entity(record)
            doc_id = f"film_{record_id(record)}"
            evidence.append(line_evidence(doc_id, documents[doc_id], "导演：", [director["canonical_id"], movie["canonical_id"]]))
            evidence.append(line_evidence(doc_id, documents[doc_id], "演员：", [actor_id, movie["canonical_id"]]))
            actor_paths.append({"nodes": [director["canonical_id"], movie["canonical_id"], actor_id], "edges": [edge(director, "directed", movie), edge(actor, "acted_in", movie)]})
        paths.append({"answer_entity_id": actor_id, "distinct_film_paths": actor_paths})
    dedup: dict[tuple[str, int, int], dict[str, Any]] = {}
    for item in evidence:
        key = (item["doc_id"], item["char_start"], item["char_end"])
        if key not in dedup:
            dedup[key] = dict(item)
        else:
            dedup[key]["supports"] = sorted(set(dedup[key]["supports"]) | set(item["supports"]))
    return {
        "schema_version": 2, "id": qid, "split": split, "type": "list", "answer_kind": "entity_set",
        "question": f"哪些演员在{director['name']}执导的影片中出现过不止一次？",
        "reference_answer": "；".join(item["name"] for item in gold), "gold_answers": gold,
        "answer_candidates": sorted(candidates_by_id.values(), key=lambda item: item["name"]), "answerable": True,
        "gold_documents": sorted({item["doc_id"] for item in dedup.values()}),
        "gold_evidence": sorted(dedup.values(), key=lambda item: (item["doc_id"], item["char_start"])),
        "required_relation_paths": paths,
        "notes": "每部影片演员先按规范名去重；答案演员必须出现在该导演至少两部不同影片中。",
    }


def make_no_answer(split: str, qid: str, question: str) -> dict[str, Any]:
    return {
        "schema_version": 2, "id": qid, "split": split, "type": "no_answer", "answer_kind": "abstention",
        "question": question, "reference_answer": "无法根据冻结语料确定", "gold_answers": [],
        "answer_candidates": [], "answerable": False, "gold_documents": [], "gold_evidence": [],
        "required_relation_paths": [], "notes": "人工构造且已核对冻结语料中不存在相关实体；用于评测可信拒答。",
    }


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    records = read_jsonl(args.source / "films.jsonl")
    by_id = {record_id(record): record for record in records}
    catalog = load_people_catalog(args.source)
    required_ids = {film_id for specs in FACT_SPECS.values() for _, film_id in specs}
    required_ids.update(film_id for specs in PATH_SPECS.values() for _, film_id, _, _ in specs)
    for specs in AGGREGATE_SPECS.values():
        for _, director_id in specs:
            _, films, _ = aggregate_for_director(records, director_id, catalog)
            required_ids.update(record_id(record) for record in films)
    if len(required_ids) > args.size:
        raise SystemExit(f"Curated evidence requires {len(required_ids)} documents, above requested size {args.size}.")
    remaining = [record for record in records if record_id(record) not in required_ids]
    random.Random(args.seed).shuffle(remaining)
    chosen_ids = sorted(required_ids) + [record_id(record) for record in remaining[: args.size - len(required_ids)]]
    selected = [by_id[item] for item in chosen_ids]
    plan = {"version": VERSION, "documents": len(selected), "required_evidence_documents": len(required_ids), "dev_questions": 5, "test_questions": 20, "output": str(args.out)}
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    documents_dir = args.out / "documents"
    documents_dir.mkdir(parents=True, exist_ok=True)
    documents: dict[str, str] = {}
    manifest_rows = []
    for record in selected:
        doc_id = f"film_{record_id(record)}"
        text = compact_document(record)
        documents[doc_id] = text
        filename = f"{doc_id}.txt"
        (documents_dir / filename).write_text(text, encoding="utf-8")
        manifest_rows.append({
            "doc_id": doc_id, "title": title(record), "path": f"documents/{filename}",
            "source": record.get("source_document", {}).get("url", ""), "revision_id": record.get("source_document", {}).get("revision_id"),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "tags": ["film", "wikipedia", "deduplicated-structured-facts"],
        })
    manifest_path = args.out / "manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows), encoding="utf-8")

    movie_candidates = list({movie_entity(record)["canonical_id"]: movie_entity(record) for record in selected}.values())
    director_map: dict[str, dict[str, Any]] = {}
    for record in selected:
        for raw in unique_people(record.get("directors", [])):
            item = person_entity(raw, catalog)
            director_map[item["canonical_id"]] = item
    director_candidates = sorted(director_map.values(), key=lambda item: item["name"])
    question_sets: dict[str, list[dict[str, Any]]] = {"dev": [], "test": []}
    for split in ("dev", "test"):
        for qid, film_id in FACT_SPECS[split]:
            question_sets[split].append(make_fact(split, qid, by_id[film_id], documents, catalog, director_candidates))
        for qid, film_id, left, right in PATH_SPECS[split]:
            question_sets[split].append(make_path(split, qid, by_id[film_id], left, right, documents, catalog, movie_candidates))
        for qid, director_id in AGGREGATE_SPECS[split]:
            question_sets[split].append(make_aggregate(split, qid, records, director_id, documents, catalog))
        for qid, question in NO_ANSWER_SPECS[split]:
            question_sets[split].append(make_no_answer(split, qid, question))
        path = args.out / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in question_sets[split]), encoding="utf-8")

    manifest_ids = {row["doc_id"] for row in manifest_rows}
    for questions in question_sets.values():
        for question in questions:
            if set(question["gold_documents"]) - manifest_ids:
                raise AssertionError(f"{question['id']} refers to a document outside the frozen corpus")
            for evidence in question["gold_evidence"]:
                actual = documents[evidence["doc_id"]][evidence["char_start"]:evidence["char_end"]]
                if actual != evidence["quote"]:
                    raise AssertionError(f"Evidence offset mismatch in {question['id']}")
            if question["answer_kind"] == "entity_set":
                for path in question["required_relation_paths"]:
                    film_ids = {item["nodes"][1] for item in path["distinct_film_paths"]}
                    if len(film_ids) < 2:
                        raise AssertionError(f"{question['id']} contains a single-film pseudo aggregate")

    summary = {
        **plan, "seed": args.seed, "source": str(args.source),
        "rendering": "Wikipedia intro + deduplicated explicit metadata facts",
        "manual_review": "curated question targets; deterministic evidence/ID/path validation",
        "known_limitations": ["Wikipedia-oriented rather than reviewer-attributed opinion corpus", "no-answer items use synthetic absent names"],
    }
    benchmark_path = args.out / "benchmark.json"
    benchmark_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    frozen_files = [manifest_path, args.out / "dev.jsonl", args.out / "test.jsonl", benchmark_path, *sorted(documents_dir.glob("*.txt"))]
    freeze = {"version": VERSION, "files": {str(path.relative_to(args.out)): file_sha256(path) for path in frozen_files}}
    (args.out / "freeze_manifest.json").write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**summary, "frozen_files": len(frozen_files)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
