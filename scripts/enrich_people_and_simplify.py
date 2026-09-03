#!/usr/bin/env python3
"""Add standalone actor/director entries and convert the corpus to Simplified Chinese."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from opencc import OpenCC

from fetch_wikipedia_300_films import (
    LICENSE,
    LICENSE_URL,
    derive_collaborations,
    flatten_relations,
    now_iso,
    request_json,
    write_jsonl,
)


CONVERTER = OpenCC("t2s")
PERSON_FIELDS = {"actors": "演员", "directors": "导演"}


def chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    for index in range(0, len(items), size):
        yield items[index:index + size]


def simplify_text(value: str) -> str:
    for _ in range(5):
        converted = CONVERTER.convert(value)
        if converted == value:
            break
        value = converted
    return value


def simplify_object(value: Any) -> Any:
    if isinstance(value, str):
        return simplify_text(value)
    if isinstance(value, list):
        return [simplify_object(item) for item in value]
    if isinstance(value, dict):
        return {simplify_text(str(key)): simplify_object(item) for key, item in value.items()}
    return value


def normalize_title(value: str) -> str:
    return value.replace("_", " ").split("#", 1)[0].strip()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect_people(films: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    people: dict[str, dict[str, Any]] = {}
    for film in films:
        film_ref = {"id": film["film"]["id"], "name": film["film"]["name"]}
        for item in film[field]:
            person = people.setdefault(
                item["id"],
                {
                    "old_id": item["id"], "names": set(), "wiki_titles": set(),
                    "films": {}, "roles_by_film": defaultdict(set),
                },
            )
            if item.get("name"):
                person["names"].add(item["name"])
            if item.get("wiki_title"):
                person["wiki_titles"].add(normalize_title(item["wiki_title"]))
            person["films"][film_ref["id"]] = film_ref
            for role in item.get("roles", []):
                person["roles_by_film"][film_ref["id"]].add(role)
    return people


def load_cache(cache_path: Path) -> dict[str, dict[str, Any] | None]:
    cache: dict[str, dict[str, Any] | None] = {}
    if not cache_path.exists():
        return cache
    for row in load_jsonl(cache_path):
        cache[row["requested_title"]] = None if row.get("missing") else row["page"]
    return cache


def append_cache(cache_path: Path, rows: list[dict[str, Any]]) -> None:
    with cache_path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_title(requested: str, forward: dict[str, str]) -> str:
    current = normalize_title(requested)
    seen: set[str] = set()
    while current in forward and current not in seen:
        seen.add(current)
        current = normalize_title(forward[current])
    return current


def fetch_person_batch(batch: list[str]) -> list[dict[str, Any]]:
    data = request_json(
        {
            "action": "query", "format": "json", "titles": "|".join(batch),
            "redirects": "1", "converttitles": "1", "variant": "zh-cn",
            "prop": "extracts|revisions|info|pageprops|categories",
            "explaintext": "1", "exintro": "1", "exlimit": str(len(batch)),
            "rvprop": "ids|timestamp|content", "rvslots": "main",
            "inprop": "url", "cllimit": "max", "utf8": "1",
        }
    )
    query = data.get("query", {})
    forward: dict[str, str] = {}
    for key in ("normalized", "converted", "redirects"):
        for item in query.get(key, []):
            forward[normalize_title(item["from"])] = normalize_title(item["to"])

    pages_by_title: dict[str, dict[str, Any]] = {}
    for page in query.get("pages", {}).values():
        if "missing" in page:
            continue
        revision = (page.get("revisions") or [{}])[0]
        slot = revision.get("slots", {}).get("main", {})
        page_title = normalize_title(page.get("title", ""))
        pages_by_title[page_title] = {
            "pageid": page.get("pageid"),
            "wikidata_qid": page.get("pageprops", {}).get("wikibase_item"),
            "title": page_title,
            "url": page.get("fullurl"),
            "revision_id": revision.get("revid"),
            "revision_timestamp": revision.get("timestamp"),
            "intro": re.sub(r"\s+", " ", page.get("extract", "")).strip(),
            "raw_wikitext": slot.get("content", slot.get("*", "")),
            "categories": [
                item.get("title", "").replace("Category:", "")
                for item in page.get("categories", [])
            ],
        }

    rows: list[dict[str, Any]] = []
    for requested in batch:
        resolved = resolve_title(requested, forward)
        page = pages_by_title.get(resolved)
        rows.append({"requested_title": requested, "missing": page is None, "page": page})
    return rows


def fetch_person_pages(
    requested_titles: list[str], cache_path: Path, batch_size: int, workers: int,
) -> dict[str, dict[str, Any] | None]:
    cache = load_cache(cache_path)
    pending = [title for title in requested_titles if title not in cache]
    total = (len(pending) + batch_size - 1) // batch_size
    print(f"Person pages cached: {len(cache)}; pending: {len(pending)}", flush=True)

    pending_batches = list(chunks(pending, batch_size))
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch_person_batch, batch) for batch in pending_batches]
        for future in as_completed(futures):
            cache_rows = future.result()
            for row in cache_rows:
                cache[row["requested_title"]] = None if row.get("missing") else row["page"]
            append_cache(cache_path, cache_rows)
            completed += 1
            print(f"Person page batches: {completed}/{total}; processed: {len(cache)}", flush=True)
    return cache


def choose_page(person: dict[str, Any], page_cache: dict[str, dict[str, Any] | None]) -> dict[str, Any] | None:
    for title in sorted(person["wiki_titles"]):
        page = page_cache.get(title)
        if page:
            return page
    return None


def build_person_records(
    people: dict[str, dict[str, Any]], entity_type: str,
    page_cache: dict[str, dict[str, Any] | None],
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, str]]:
    records: list[dict[str, Any]] = []
    id_map: dict[str, str] = {}
    name_map: dict[str, str] = {}
    for old_id, person in people.items():
        page = choose_page(person, page_cache)
        aliases = sorted({simplify_text(name) for name in person["names"] if name})
        page_title = simplify_text(page["title"]) if page else None
        display_name = page_title or (aliases[0] if aliases else simplify_text(old_id))
        if page and page.get("wikidata_qid"):
            person_id = page["wikidata_qid"]
        elif page:
            person_id = f"WP:{page['pageid']}"
        elif old_id.startswith("Q"):
            person_id = old_id
        else:
            person_id = simplify_text(old_id)
        id_map[old_id] = person_id
        name_map[old_id] = display_name

        filmography = []
        for film_id, film in person["films"].items():
            item = {"id": film_id, "name": simplify_text(film["name"])}
            roles = sorted(simplify_text(role) for role in person["roles_by_film"].get(film_id, set()))
            if entity_type == "演员":
                item["roles"] = roles
            filmography.append(item)
        filmography.sort(key=lambda item: item["name"])

        source_document = None
        if page:
            source_document = {
                "title": page_title,
                "intro": simplify_text(page["intro"]),
                "raw_wikitext": simplify_text(page["raw_wikitext"]),
                "categories": [simplify_text(item) for item in page["categories"]],
                "url": page["url"], "pageid": page["pageid"],
                "revision_id": page["revision_id"],
                "revision_timestamp": page["revision_timestamp"],
                "source": "中文维基百科", "license": LICENSE,
                "license_url": LICENSE_URL, "fetched_at": now_iso(),
            }
        records.append(
            {
                "person": {
                    "id": person_id, "name": display_name,
                    "entity_type": entity_type, "aliases": aliases,
                },
                "filmography": filmography,
                "related_film_count": len(filmography),
                "has_wikipedia_text": page is not None,
                "source_document": source_document,
            }
        )
    records.sort(key=lambda row: row["person"]["name"])
    return records, id_map, name_map


def update_film_people(
    films: list[dict[str, Any]], field: str,
    id_map: dict[str, str], name_map: dict[str, str],
) -> None:
    for film in films:
        for person in film[field]:
            old_id = person["id"]
            person["id"] = id_map.get(old_id, simplify_text(old_id))
            person["name"] = name_map.get(old_id, simplify_text(person["name"]))


def write_person_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [
            "person_id", "name", "entity_type", "related_film_count",
            "has_wikipedia_text", "intro_chars", "wikitext_chars", "source_url",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            document = record.get("source_document") or {}
            writer.writerow(
                {
                    "person_id": record["person"]["id"],
                    "name": record["person"]["name"],
                    "entity_type": record["person"]["entity_type"],
                    "related_film_count": record["related_film_count"],
                    "has_wikipedia_text": record["has_wikipedia_text"],
                    "intro_chars": len(document.get("intro", "")),
                    "wikitext_chars": len(document.get("raw_wikitext", "")),
                    "source_url": document.get("url", ""),
                }
            )


def write_film_manifest(path: Path, films: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [
            "film_id", "title", "directors", "actor_count", "screenwriters",
            "production_companies", "genres", "award_count", "nomination_count",
            "role_count", "intro_chars", "wikitext_chars", "source_url",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in films:
            writer.writerow(
                {
                    "film_id": record["film"]["id"], "title": record["film"]["name"],
                    "directors": "|".join(item["name"] for item in record["directors"]),
                    "actor_count": len(record["actors"]),
                    "screenwriters": "|".join(item["name"] for item in record["screenwriters"]),
                    "production_companies": "|".join(item["name"] for item in record["production_companies"]),
                    "genres": "|".join(item["name"] for item in record["genres"]),
                    "award_count": len(record["awards_received"]),
                    "nomination_count": len(record["nominations"]),
                    "role_count": sum(len(item.get("roles", [])) for item in record["actors"]),
                    "intro_chars": len(record["source_document"]["intro"]),
                    "wikitext_chars": len(record["source_document"]["raw_wikitext"]),
                    "source_url": record["source_document"]["url"],
                }
            )


def normalize_existing_files(data_dir: Path) -> None:
    names = [
        "films.jsonl", "actors.jsonl", "directors.jsonl", "relations.jsonl",
        "collaborations.jsonl", "manifest.csv", "actor_manifest.csv",
        "director_manifest.csv", "stats.json",
    ]
    for name in names:
        path = data_dir / name
        if not path.exists():
            continue
        encoding = "utf-8-sig" if path.suffix == ".csv" else "utf-8"
        original = path.read_text(encoding=encoding)
        normalized = simplify_text(original)
        path.write_text(normalized, encoding=encoding, newline="")
        print(f"Normalized: {name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/wikipedia_300_films_final")
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--normalize-only", action="store_true")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    if args.normalize_only:
        normalize_existing_files(data_dir)
        return
    films = load_jsonl(data_dir / "films.jsonl")

    actors = collect_people(films, "actors")
    directors = collect_people(films, "directors")
    requested_titles = sorted(
        {title for person in list(actors.values()) + list(directors.values()) for title in person["wiki_titles"]}
    )
    cache_path = data_dir / ".person_pages_cache.jsonl"
    page_cache = fetch_person_pages(
        requested_titles, cache_path, max(1, min(args.batch_size, 20)),
        max(1, min(args.workers, 4)),
    )

    actor_records, actor_ids, actor_names = build_person_records(actors, "演员", page_cache)
    director_records, director_ids, director_names = build_person_records(directors, "导演", page_cache)
    update_film_people(films, "actors", actor_ids, actor_names)
    update_film_people(films, "directors", director_ids, director_names)
    films = simplify_object(films)
    relations = flatten_relations(films)
    collaborations = derive_collaborations(films)

    write_jsonl(data_dir / "films.jsonl", films)
    write_jsonl(data_dir / "actors.jsonl", actor_records)
    write_jsonl(data_dir / "directors.jsonl", director_records)
    write_jsonl(data_dir / "relations.jsonl", relations)
    write_jsonl(data_dir / "collaborations.jsonl", collaborations)
    write_film_manifest(data_dir / "manifest.csv", films)
    write_person_manifest(data_dir / "actor_manifest.csv", actor_records)
    write_person_manifest(data_dir / "director_manifest.csv", director_records)

    previous_stats_path = data_dir / "stats.json"
    previous_stats = json.loads(previous_stats_path.read_text(encoding="utf-8"))
    stats = {
        **previous_stats,
        "generated_at": now_iso(),
        "language_variant": "简体中文（OpenCC t2s）",
        "actor_records": len(actor_records),
        "actor_records_with_wikipedia_text": sum(row["has_wikipedia_text"] for row in actor_records),
        "director_records": len(director_records),
        "director_records_with_wikipedia_text": sum(row["has_wikipedia_text"] for row in director_records),
        "relation_rows": len(relations),
        "collaboration_pairs": len(collaborations),
        "entity_files": {
            "影片": "films.jsonl", "演员": "actors.jsonl", "导演": "directors.jsonl",
        },
    }
    previous_stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    cache_path.unlink(missing_ok=True)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
