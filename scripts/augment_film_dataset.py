#!/usr/bin/env python3
"""Clean the bulk film corpus, add curated classics, and enrich sparse fields."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any

from fetch_wikipedia_300_films import (
    LICENSE,
    LICENSE_URL,
    build_records,
    clean_markup,
    derive_collaborations,
    flatten_relations,
    now_iso,
    request_json,
    schema_from_wikitext,
    write_jsonl,
)


CURATED_FILMS = [
    "追随", "记忆碎片", "失眠症 (2002年电影)", "蝙蝠侠：侠影之谜", "致命魔术", "黑暗骑士",
    "盗梦空间", "黑暗骑士崛起", "星际穿越", "敦刻尔克 (电影)", "信条 (电影)", "奥本海默 (电影)",
    "泰坦尼克号 (1997年电影)", "阿凡达 (电影)", "终结者2：审判日", "异形2", "侏罗纪公园",
    "辛德勒的名单", "拯救大兵瑞恩", "少数派报告 (电影)", "华尔街之狼", "无间道风云",
    "出租车司机", "好家伙", "爱尔兰人 (2019年电影)", "低俗小说", "杀死比尔", "无耻混蛋",
    "被解救的姜戈", "搏击俱乐部", "七宗罪 (电影)", "社交网络 (电影)", "银翼杀手",
    "角斗士 (电影)", "火星救援 (电影)", "异形 (电影)", "降临 (电影)", "银翼杀手2049",
    "沙丘 (2021年电影)", "沙丘2", "布达佩斯大饭店", "月升王国", "寄生虫 (电影)", "杀人回忆",
    "母亲 (2009年电影)", "雪国列车", "老男孩 (2003年电影)", "小姐 (2016年电影)", "千与千寻",
    "龙猫", "哈尔的移动城堡", "幽灵公主", "七武士", "罗生门", "卧虎藏龙", "断背山",
    "少年派的奇幻漂流 (电影)", "花样年华", "重庆森林", "2046 (电影)", "无间道", "英雄 (2002年电影)",
    "红高粱 (电影)", "霸王别姬 (电影)", "活着 (电影)", "让子弹飞", "阳光灿烂的日子",
    "疯狂的石头", "无人区 (电影)", "天下无贼", "芳华 (电影)", "一代宗师", "色，戒",
    "饮食男女", "悲情城市", "牯岭街少年杀人事件", "功夫 (电影)", "少林足球", "英雄本色",
    "甜蜜蜜 (电影)", "花束般的恋爱",
]

GENRE_PATTERNS = {
    "动作": r"動作|动作|武打|武俠|武侠",
    "剧情": r"劇情|剧情",
    "喜剧": r"喜劇|喜剧",
    "爱情": r"愛情|爱情|浪漫",
    "科幻": r"科幻",
    "恐怖": r"恐怖",
    "惊悚": r"驚悚|惊悚",
    "犯罪": r"犯罪",
    "悬疑": r"懸疑|悬疑|推理",
    "动画": r"動畫|动画",
    "纪录": r"紀錄片|纪录片|記錄片",
    "奇幻": r"奇幻|魔幻",
    "冒险": r"冒險|冒险",
    "战争": r"戰爭|战争",
    "传记": r"傳記|传记",
    "音乐": r"音樂|音乐|歌舞",
    "家庭": r"家庭",
    "灾难": r"災難|灾难",
    "西部": r"西部",
    "历史": r"歷史|历史",
    "体育": r"體育|体育|運動|运动",
}

AWARD_TERMS = r"(?:獎|奖|奧斯卡|奥斯卡|金球|金馬|金马|金像|影展|電影節|电影节|影評人|影评人)"
JUNK_TITLE = re.compile(
    r"^\d{4}年(?:電影|电影)$|角逐名單|角逐名单|電影業的影響|电影业的影响|^CNN\+$|^Disney\+$|^Star\+$"
)


def fetch_curated_pages(titles: list[str], batch_size: int = 15) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    for index in range(0, len(titles), batch_size):
        batch = titles[index:index + batch_size]
        data = request_json(
            {
                "action": "query", "format": "json", "titles": "|".join(batch), "redirects": "1",
                "prop": "extracts|revisions|info|pageprops|categories",
                "explaintext": "1", "exintro": "1", "exlimit": str(batch_size),
                "rvprop": "ids|timestamp|content", "rvslots": "main", "inprop": "url",
                "cllimit": "max", "utf8": "1",
            }
        )
        for page in data.get("query", {}).get("pages", {}).values():
            if "missing" in page:
                continue
            revision = (page.get("revisions") or [{}])[0]
            slot = revision.get("slots", {}).get("main", {})
            pages.append(
                {
                    "pageid": int(page["pageid"]), "title": page.get("title", ""),
                    "url": page.get("fullurl", ""),
                    "wikidata_qid": page.get("pageprops", {}).get("wikibase_item"),
                    "revision_id": revision.get("revid"), "revision_timestamp": revision.get("timestamp"),
                    "intro": re.sub(r"\s+", " ", page.get("extract", "")).strip(),
                    "raw_wikitext": slot.get("content", slot.get("*", "")),
                    "categories": [cat.get("title", "").replace("Category:", "") for cat in page.get("categories", [])],
                }
            )
        print(f"Curated page batches: {index // batch_size + 1}/{(len(titles) + batch_size - 1) // batch_size}", flush=True)
        time.sleep(0.3)
    return pages


def record_from_page(page: dict[str, Any]) -> dict[str, Any]:
    schema, raw_fields = schema_from_wikitext(page["raw_wikitext"])
    return {
        "film": {
            "id": page.get("wikidata_qid") or f"WP:{page['pageid']}",
            "name": page["title"], "entity_type": "影片", "year_category": "精选经典影片",
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


def entity(name: str, raw: str) -> dict[str, Any]:
    return {"id": f"NAME:{name}", "name": name, "wiki_title": None, "raw": raw}


def enrich_genres(record: dict[str, Any]) -> None:
    existing = {item["name"] for item in record["genres"]}
    evidence = " | ".join(record["source_document"].get("categories", []))
    evidence += " | " + str(record["source_document"].get("raw_infobox_fields", {}).get("genre", ""))
    for genre, pattern in GENRE_PATTERNS.items():
        if genre not in existing and re.search(pattern, evidence, re.I):
            record["genres"].append(entity(genre, evidence))


def enrich_awards(record: dict[str, Any]) -> None:
    wikitext = record["source_document"]["raw_wikitext"]
    categories = record["source_document"].get("categories", [])
    received = {item["name"] for item in record["awards_received"]}
    nominated = {item["name"] for item in record["nominations"]}

    for category in categories:
        if re.search(AWARD_TERMS, category) and re.search(r"獲|获|得主|最佳|金牌|銀牌|银牌", category):
            name = clean_markup(category)
            if name and name not in received:
                record["awards_received"].append(entity(name, f"Category:{category}"))
                received.add(name)

    section = re.search(
        r"^==+\s*(?:獎項|奖项|榮譽|荣誉|提名|得獎|获奖)[^=]*==+\s*$([\s\S]*?)(?=^==[^=]|\Z)",
        wikitext, re.M,
    )
    if not section:
        return
    for line in section.group(1).splitlines():
        if not re.search(AWARD_TERMS, line):
            continue
        candidates = re.findall(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|([^\]]+))?\]\]", line)
        award_names = [clean_markup(label or target) for target, label in candidates if re.search(AWARD_TERMS, label or target)]
        if "提名" in line:
            for name in award_names:
                if name and name not in nominated:
                    record["nominations"].append(entity(name, line.strip()))
                    nominated.add(name)
        if re.search(r"獲獎|获奖|得獎|得奖|勝出|胜出|贏得|赢得|winner", line, re.I):
            for name in award_names:
                if name and name not in received:
                    record["awards_received"].append(entity(name, line.strip()))
                    received.add(name)


def valid_film(record: dict[str, Any]) -> bool:
    title = record["film"]["name"]
    if JUNK_TITLE.search(title):
        return False
    if record["directors"] or record["actors"] or record["screenwriters"]:
        return True
    categories = " ".join(record["source_document"].get("categories", []))
    return bool(re.search(r"電影作品|电影作品|電影|电影|影片|動畫電影|动画电影", categories))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/wikipedia_300_films/films.jsonl")
    parser.add_argument("--out-dir", default="data/wikipedia_300_films_final")
    args = parser.parse_args()
    input_path = Path(args.input)
    records = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [record for record in records if valid_film(record)]
    existing_titles = {record["film"]["name"] for record in records}

    curated_pages = fetch_curated_pages(CURATED_FILMS)
    for page in curated_pages:
        if page["title"] not in existing_titles and page["raw_wikitext"]:
            record = record_from_page(page)
            if valid_film(record):
                records.append(record)
                existing_titles.add(page["title"])

    for record in records:
        enrich_genres(record)
        enrich_awards(record)

    if len(records) < 300:
        raise RuntimeError(f"Only {len(records)} valid film records remain")
    relations = flatten_relations(records)
    collaborations = derive_collaborations(records)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    field_names = [
        "directors", "actors", "screenwriters", "production_companies", "awards_received",
        "nominations", "genres", "adapted_from", "previous_works", "sequels",
    ]
    stats = {
        "generated_at": now_iso(), "film_records": len(records),
        "full_wikitext_records": sum(bool(row["source_document"]["raw_wikitext"]) for row in records),
        "intro_records": sum(bool(row["source_document"]["intro"]) for row in records),
        "relation_rows": len(relations), "collaboration_pairs": len(collaborations),
        "field_coverage": {field: sum(bool(row[field]) for row in records) for field in field_names},
        "films_with_character_roles": sum(any(actor.get("roles") for actor in row["actors"]) for row in records),
        "meets_300_film_requirement": len(records) >= 300,
        "schema": ["影片", "导演", "演员", "编剧", "制片公司", "奖项", "类型", "角色"],
        "relations": ["执导", "出演", "编剧", "出品", "获奖", "提名", "改编自", "前作", "续作", "合作"],
    }
    write_jsonl(out_dir / "films.jsonl", records)
    write_jsonl(out_dir / "relations.jsonl", relations)
    write_jsonl(out_dir / "collaborations.jsonl", collaborations)
    with (out_dir / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [
            "film_id", "title", "directors", "actor_count", "screenwriters",
            "production_companies", "genres", "award_count", "nomination_count",
            "role_count", "intro_chars", "wikitext_chars", "source_url",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "film_id": record["film"]["id"],
                    "title": record["film"]["name"],
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
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "README.txt").write_text(
        "films.jsonl: one valid film per line, including full source text and rough Schema extraction.\n"
        "relations.jsonl: flattened graph edges with evidence URL and raw source fragment.\n"
        "collaborations.jsonl: derived person-to-person collaborations with counts and shared films.\n"
        "Structured extraction is intentionally rough; raw_wikitext is retained for later cleaning.\n"
        "Chinese Wikipedia text is CC BY-SA 4.0; retain page URL and revision attribution.\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
