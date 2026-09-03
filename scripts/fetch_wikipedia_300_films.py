#!/usr/bin/env python3
"""Bulk-fetch 300+ Chinese Wikipedia film pages and roughly map the KG schema."""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


API_URL = "https://zh.wikipedia.org/w/api.php"
USER_AGENT = "FilmKGRAGCourseProject/3.0 (educational bulk corpus builder)"
LICENSE = "CC BY-SA 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
YEAR_CATEGORIES = [f"Category:{year}年電影" for year in range(2024, 2004, -1)]

FIELD_ALIASES = {
    "directors": {"director", "directors", "导演", "導演"},
    "actors": {"starring", "主演", "演员", "演員", "actor", "actors"},
    "screenwriters": {"writer", "writers", "screenplay", "screenwriter", "编剧", "編劇"},
    "production_companies": {"studio", "productioncompany", "productioncompanies", "制片公司", "製片公司", "出品公司"},
    "awards_received": {"awards", "award", "获奖", "獲獎"},
    "nominations": {"nominations", "nomination", "提名"},
    "genres": {"genre", "genres", "类型", "類型", "片种", "片種"},
    "adapted_from": {"basedon", "based_on", "原作", "改编自", "改編自"},
    "previous_works": {"precededby", "preceded_by", "前作"},
    "sequels": {"followedby", "followed_by", "sequel", "续作", "續作"},
}

RELATION_NAMES = {
    "directors": "执导",
    "actors": "出演",
    "screenwriters": "编剧",
    "production_companies": "出品",
    "awards_received": "获奖",
    "nominations": "提名",
    "genres": "类型",
    "adapted_from": "改编自",
    "previous_works": "前作",
    "sequels": "续作",
}


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def request_json(params: dict[str, Any], retries: int = 6) -> dict[str, Any]:
    request_url = f"{API_URL}?{urllib.parse.urlencode(params, doseq=True)}"
    request = urllib.request.Request(request_url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            retry_after = 0
            if isinstance(exc, urllib.error.HTTPError):
                value = exc.headers.get("Retry-After", "")
                retry_after = int(value) if value.isdigit() else 0
                if exc.code not in (429, 500, 502, 503, 504):
                    break
            time.sleep(min(max(retry_after, 2**attempt), 30))
    raise RuntimeError(f"MediaWiki request failed: {last_error}") from last_error


def chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    for index in range(0, len(items), size):
        yield items[index:index + size]


def collect_catalog(target: int) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    seen_pageids: set[int] = set()
    visited_categories: set[str] = set()
    queue: list[tuple[str, int]] = [(category, 0) for category in YEAR_CATEGORIES]
    while queue and len(catalog) < target:
        category, depth = queue.pop(0)
        if category in visited_categories:
            continue
        visited_categories.add(category)
        continuation: str | None = None
        while len(catalog) < target:
            params: dict[str, Any] = {
                "action": "query", "format": "json", "list": "categorymembers",
                "cmtitle": category, "cmnamespace": "0|14", "cmtype": "page|subcat",
                "cmlimit": "500", "utf8": "1",
            }
            if continuation:
                params["cmcontinue"] = continuation
            data = request_json(params)
            members = data.get("query", {}).get("categorymembers", [])
            for member in members:
                if int(member.get("ns", -1)) == 14:
                    if depth < 2:
                        queue.append((member["title"], depth + 1))
                    continue
                pageid = int(member["pageid"])
                if pageid not in seen_pageids:
                    seen_pageids.add(pageid)
                    catalog.append(
                        {
                            "pageid": pageid,
                            "title": member["title"],
                            "category": category.replace("Category:", ""),
                        }
                    )
                    if len(catalog) >= target:
                        break
            continuation = data.get("continue", {}).get("cmcontinue")
            if not continuation or not members:
                break
        print(f"Catalog after {category} (depth {depth}): {len(catalog)} films", flush=True)
        time.sleep(0.2)
    return catalog


def fetch_pages(catalog: list[dict[str, Any]], batch_size: int) -> dict[int, dict[str, Any]]:
    pages: dict[int, dict[str, Any]] = {}
    total = (len(catalog) + batch_size - 1) // batch_size
    for index, batch in enumerate(chunks(catalog, batch_size), start=1):
        data = request_json(
            {
                "action": "query", "format": "json",
                "pageids": "|".join(str(item["pageid"]) for item in batch),
                "prop": "extracts|revisions|info|pageprops|categories",
                "explaintext": "1", "exintro": "1", "exlimit": str(batch_size),
                "rvprop": "ids|timestamp|content", "rvslots": "main",
                "inprop": "url", "cllimit": "max", "utf8": "1",
            }
        )
        for page in data.get("query", {}).get("pages", {}).values():
            if "missing" in page:
                continue
            revision = (page.get("revisions") or [{}])[0]
            slot = revision.get("slots", {}).get("main", {})
            pages[int(page["pageid"])] = {
                "pageid": int(page["pageid"]),
                "title": page.get("title", ""),
                "url": page.get("fullurl", ""),
                "wikidata_qid": page.get("pageprops", {}).get("wikibase_item"),
                "revision_id": revision.get("revid"),
                "revision_timestamp": revision.get("timestamp"),
                "intro": re.sub(r"\s+", " ", page.get("extract", "")).strip(),
                "raw_wikitext": slot.get("content", slot.get("*", "")),
                "categories": [item.get("title", "").replace("Category:", "") for item in page.get("categories", [])],
            }
        print(f"Page batches: {index}/{total}; downloaded: {len(pages)}", flush=True)
        time.sleep(0.3)
    return pages


def normalize_key(value: str) -> str:
    return re.sub(r"[\s_-]+", "", value.strip().casefold())


def extract_infobox(wikitext: str) -> str:
    match = re.search(r"\{\{\s*(?:infobox\s*film|電影資訊框|电影信息框|電影信息|电影信息)", wikitext, re.I)
    if not match:
        return ""
    start = match.start()
    depth = 0
    index = start
    while index < len(wikitext) - 1:
        pair = wikitext[index:index + 2]
        if pair == "{{":
            depth += 1
            index += 2
            continue
        if pair == "}}":
            depth -= 1
            index += 2
            if depth == 0:
                return wikitext[start:index]
            continue
        index += 1
    return wikitext[start:]


def parse_infobox_fields(infobox: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    current_key: str | None = None
    buffer: list[str] = []
    for line in infobox.splitlines()[1:]:
        match = re.match(r"^\s*\|\s*([^=]+?)\s*=\s*(.*)$", line)
        if match:
            if current_key:
                fields[current_key] = "\n".join(buffer).strip()
            current_key = normalize_key(match.group(1))
            buffer = [match.group(2)]
        elif current_key:
            buffer.append(line)
    if current_key:
        fields[current_key] = "\n".join(buffer).strip()
    return fields


def clean_markup(value: str) -> str:
    value = re.sub(r"<!--.*?-->", "", value, flags=re.S)
    value = re.sub(r"<ref\b[^>]*>.*?</ref>|<ref\b[^>]*/>", "", value, flags=re.I | re.S)
    value = re.sub(r"\{\{(?:ubl|unbulleted list|plainlist|plain list|flatlist|flat list)\s*\|", "", value, flags=re.I)
    value = re.sub(r"\{\{[^{}]*\}\}", "", value)
    value = re.sub(r"\[\[(?:[^\]|]+\|)?([^\]]+)\]\]", r"\1", value)
    value = re.sub(r"<br\s*/?>", "、", value, flags=re.I)
    value = re.sub(r"</?[^>]+>", "", value)
    value = value.replace("'''", "").replace("''", "")
    return re.sub(r"\s+", " ", value).strip(" |,，、;；\n")


def entities_from_value(value: str) -> list[dict[str, Any]]:
    links = re.findall(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|([^\]]+))?\]\]", value)
    entities: list[dict[str, Any]] = []
    seen: set[str] = set()
    for target, label in links:
        name = clean_markup(label or target)
        target = target.strip()
        if not name or target.lower().startswith(("file:", "image:", "category:")):
            continue
        entity_id = f"WP:{target.replace(' ', '_')}"
        if entity_id not in seen:
            seen.add(entity_id)
            entities.append({"id": entity_id, "name": name, "wiki_title": target, "raw": value})
    if entities:
        return entities
    for name in re.split(r"(?:<br\s*/?>|、|，|,|;|；|\n|\*)+", value, flags=re.I):
        name = clean_markup(name)
        if 1 < len(name) <= 100:
            entity_id = f"NAME:{name}"
            if entity_id not in seen:
                seen.add(entity_id)
                entities.append({"id": entity_id, "name": name, "wiki_title": None, "raw": value})
    return entities


def parse_cast_roles(wikitext: str) -> dict[str, list[str]]:
    roles: dict[str, set[str]] = defaultdict(set)
    section_match = re.search(
        r"^==+\s*(?:演員|演员|角色|主要角色|演員表|演员表|角色介绍|角色介紹)\s*==+\s*$([\s\S]*?)(?=^==[^=]|\Z)",
        wikitext, re.M,
    )
    if not section_match:
        return {}
    for line in section_match.group(1).splitlines():
        match = re.search(
            r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\].{0,20}?(?:飾演|饰演|飾|饰|as)\s*(?:\[\[([^\]|]+)(?:\|([^\]]+))?\]\]|([^\n，,；;]+))",
            line, re.I,
        )
        if match:
            actor = clean_markup(match.group(2) or match.group(1))
            role = clean_markup(match.group(4) or match.group(3) or match.group(5) or "")
            if actor and role:
                roles[actor].add(role)
    return {actor: sorted(values) for actor, values in roles.items()}


def schema_from_wikitext(wikitext: str) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    infobox = extract_infobox(wikitext)
    raw_fields = parse_infobox_fields(infobox)
    schema: dict[str, list[dict[str, Any]]] = {field: [] for field in FIELD_ALIASES}
    for field, aliases in FIELD_ALIASES.items():
        for key, raw_value in raw_fields.items():
            if key in {normalize_key(alias) for alias in aliases} and raw_value:
                schema[field].extend(entities_from_value(raw_value))

    roles = parse_cast_roles(wikitext)
    for actor in schema["actors"]:
        actor["roles"] = roles.get(actor["name"], [])
    known_actors = {actor["name"] for actor in schema["actors"]}
    for actor_name, role_names in roles.items():
        if actor_name not in known_actors:
            schema["actors"].append(
                {"id": f"NAME:{actor_name}", "name": actor_name, "wiki_title": None,
                 "raw": "cast section", "roles": role_names}
            )
    return schema, raw_fields


def build_records(catalog: list[dict[str, Any]], pages: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in catalog:
        page = pages.get(item["pageid"])
        if not page or not page["raw_wikitext"]:
            continue
        schema, raw_fields = schema_from_wikitext(page["raw_wikitext"])
        records.append(
            {
                "film": {
                    "id": page.get("wikidata_qid") or f"WP:{page['pageid']}",
                    "name": page["title"], "entity_type": "影片", "year_category": item["category"],
                },
                **schema,
                "source_document": {
                    "intro": page["intro"], "raw_wikitext": page["raw_wikitext"],
                    "raw_infobox_fields": raw_fields, "categories": page["categories"],
                    "url": page["url"], "pageid": page["pageid"],
                    "revision_id": page["revision_id"], "revision_timestamp": page["revision_timestamp"],
                    "source": "Chinese Wikipedia", "license": LICENSE, "license_url": LICENSE_URL,
                    "fetched_at": now_iso(),
                },
            }
        )
    return records


def flatten_relations(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        film = record["film"]
        for field, relation in RELATION_NAMES.items():
            for target in record[field]:
                row = {
                    "source_id": film["id"], "source_name": film["name"], "source_type": "影片",
                    "relation": relation, "target_id": target["id"], "target_name": target["name"],
                    "evidence_url": record["source_document"]["url"], "raw_evidence": target.get("raw", ""),
                }
                if field == "actors":
                    row["roles"] = target.get("roles", [])
                rows.append(row)
    return rows


def derive_collaborations(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        people: dict[str, str] = {}
        for field in ("directors", "actors", "screenwriters"):
            for person in record[field]:
                people[person["id"]] = person["name"]
        for left, right in itertools.combinations(sorted(people), 2):
            item = pairs.setdefault(
                (left, right),
                {"person_a": {"id": left, "name": people[left]},
                 "person_b": {"id": right, "name": people[right]}, "relation": "合作", "films": []},
            )
            item["films"].append(record["film"])
    for item in pairs.values():
        item["collaboration_count"] = len(item["films"])
    return sorted(pairs.values(), key=lambda row: (-row["collaboration_count"], row["person_a"]["name"]))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=330, help="Fetch extra films so 300 remain after filtering.")
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument("--out-dir", default="data/wikipedia_300_films")
    args = parser.parse_args()
    if args.count < 300:
        raise SystemExit("--count must be at least 300")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog = collect_catalog(args.count)
    if len(catalog) < 300:
        raise RuntimeError(f"Only {len(catalog)} film pages found")
    pages = fetch_pages(catalog, max(1, min(args.batch_size, 20)))
    records = build_records(catalog, pages)
    if len(records) < 300:
        raise RuntimeError(f"Only {len(records)} records contain full text")
    relations = flatten_relations(records)
    collaborations = derive_collaborations(records)

    field_coverage = {field: sum(bool(record[field]) for record in records) for field in FIELD_ALIASES}
    stats = {
        "generated_at": now_iso(), "film_records": len(records),
        "full_wikitext_records": sum(bool(row["source_document"]["raw_wikitext"]) for row in records),
        "intro_records": sum(bool(row["source_document"]["intro"]) for row in records),
        "relation_rows": len(relations), "collaboration_pairs": len(collaborations),
        "field_coverage": field_coverage, "meets_300_film_requirement": len(records) >= 300,
        "schema": ["影片", "导演", "演员", "编剧", "制片公司", "奖项", "类型", "角色"],
        "relations": ["执导", "出演", "编剧", "出品", "获奖", "提名", "改编自", "前作", "续作", "合作"],
    }
    write_jsonl(out_dir / "films.jsonl", records)
    write_jsonl(out_dir / "relations.jsonl", relations)
    write_jsonl(out_dir / "collaborations.jsonl", collaborations)
    write_jsonl(out_dir / "catalog.jsonl", catalog)
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "README.txt").write_text(
        "films.jsonl: one film per line, including rough Schema extraction and the full raw Wikipedia wikitext.\n"
        "relations.jsonl: flattened graph edges with raw evidence.\n"
        "collaborations.jsonl: collaboration counts and shared films.\n"
        "Wikipedia text is CC BY-SA 4.0. Keep each row's URL and revision metadata for attribution.\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
