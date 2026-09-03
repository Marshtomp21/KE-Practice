"""根据真实数据集自带的结构化事实生成可复现评测问题。"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

REFERENCE_FILE = PROJECT_ROOT / "eval" / "reference_facts.json"
QUESTION_FILE = PROJECT_ROOT / "eval" / "questions.yaml"
GROUND_TRUTH_FILE = PROJECT_ROOT / "eval" / "ground_truth.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-kind", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()

    if not REFERENCE_FILE.exists():
        print(f"缺少 {REFERENCE_FILE}，请先运行 scripts/import_wikipedia_films.py")
        return 1
    films = json.loads(REFERENCE_FILE.read_text(encoding="utf-8")).get("films", [])
    rng = random.Random(args.seed)
    questions: List[Dict[str, Any]] = []

    single_hop = [film for film in films if film.get("directors")]
    rng.shuffle(single_hop)
    for index, film in enumerate(single_hop[: args.per_kind], start=1):
        questions.append({
            "id": f"single_hop-{index:02d}",
            "kind": "single_hop",
            "question": f"《{film['title']}》是由谁执导的？",
            "expect": film["directors"],
            "expect_mode": "contains_any",
            "note": "来自真实 films.jsonl 的导演字段",
        })

    path_candidates = [film for film in films if len(set(film.get("actors", []))) >= 2]
    rng.shuffle(path_candidates)
    for index, film in enumerate(path_candidates[: args.per_kind], start=1):
        actors = list(dict.fromkeys(film["actors"]))
        questions.append({
            "id": f"path-{index:02d}",
            "kind": "path",
            "question": f"{actors[0]}与{actors[1]}通过哪部影片产生关联？",
            "expect": [film["title"]],
            "expect_mode": "contains_any",
            "note": "两位演员共同出现在同一真实影片记录中",
        })

    by_director: Dict[str, List[dict]] = defaultdict(list)
    for film in films:
        for director in film.get("directors", []):
            by_director[director].append(film)
    aggregate_candidates = []
    for director, directed_films in by_director.items():
        counts = Counter(actor for film in directed_films for actor in film.get("actors", []))
        repeated = sorted(actor for actor, count in counts.items() if count >= 2)
        if repeated:
            aggregate_candidates.append((director, repeated))
    rng.shuffle(aggregate_candidates)
    for index, (director, repeated) in enumerate(aggregate_candidates[: args.per_kind], start=1):
        questions.append({
            "id": f"aggregate-{index:02d}",
            "kind": "aggregate",
            "question": f"哪些演员在{director}执导的影片中出现过不止一次？",
            "expect": repeated,
            "expect_mode": "contains_any",
            "note": "由真实影片—导演—演员字段聚合得到",
        })

    QUESTION_FILE.write_text(
        yaml.safe_dump({"questions": questions}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    GROUND_TRUTH_FILE.write_text(
        json.dumps({item["id"]: item["expect"] for item in questions}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    counts = Counter(item["kind"] for item in questions)
    print(f"生成真实评测问题 {len(questions)} 道：{dict(counts)} -> {QUESTION_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
