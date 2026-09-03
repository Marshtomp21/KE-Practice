"""抽取质量评估：拿图谱和数据集自带的结构化字段对表。

两点必须说清楚，否则数字会被误读：

1. 参照不是金标准。数据集 README 自己写着结构化抽取是 rough 的，它取自 infobox，
   而我们从正文抽取。因此「参照有、我们没有」才算漏抽；「我们有、参照没有」
   不一定是错——正文里的配角、联合出品方本来就不在 infobox 里。
   所以这里对 acted_in 这类字段只把召回当作主要指标，精确率仅供参考。

2. 名称写法不一致。参照里的名字常带英文原名（「大衛·史托頓 David Stoten」），
   正文里只写中文。直接做字符串相等会大幅低估命中，因此比对前先做归一：
   繁简折叠、去空白、去英文尾巴，再允许一方是另一方的前缀。

用法：python scripts/eval_extraction.py [--samples 8]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import load_settings
from src.graph.builder import load_graph
from src.ingest.cleaner import load_variant_table

REFERENCE_FILE = PROJECT_ROOT / "eval" / "reference_facts.json"
LATIN_TAIL = re.compile(r"[A-Za-z][A-Za-z\.\-' ]*$")

FIELDS = [
    ("directed", "directors", "Person", True),
    ("acted_in", "actors", "Person", False),
    ("wrote", "screenwriters", "Person", True),
    ("produced", "companies", "Company", False),
    ("has_genre", "genres", "Genre", True),
]


def make_normalizer(table: Dict[str, str]):
    def normalize(name: str) -> str:
        folded = "".join(table.get(ch, ch) for ch in (name or ""))
        # 「中文名 English Name」只保留中文部分
        stripped = LATIN_TAIL.sub("", folded).strip()
        if len(stripped) >= 2:
            folded = stripped
        return re.sub(r"[\s·・.\-]", "", folded).lower()

    return normalize


def match_count(truth: Set[str], got: Set[str]) -> Tuple[int, Set[str]]:
    """允许前缀匹配：正文常用简称，参照常用全名。"""
    hit: Set[str] = set()
    for item in truth:
        if item in got:
            hit.add(item)
            continue
        if any(item.startswith(g) or g.startswith(item) for g in got if len(g) >= 3):
            hit.add(item)
    return len(hit), hit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=6, help="每类打印几个漏抽样例")
    args = parser.parse_args()

    if not REFERENCE_FILE.exists():
        print(f"缺少 {REFERENCE_FILE}，请先运行 scripts/import_wikipedia_films.py")
        return 1

    settings = load_settings()
    normalize = make_normalizer(load_variant_table(str(settings.path("paths.variant_table"))))
    reference = json.loads(REFERENCE_FILE.read_text(encoding="utf-8"))
    graph = load_graph(settings)

    names = {entity.id: entity.name for entity in graph.all_entities()}
    by_film: Dict[str, Dict[str, Set[str]]] = {}
    for relation in graph.all_relations():
        tail = names.get(relation.tail_id, "")
        head = names.get(relation.head_id, "")
        # 影片一侧在 has_genre 里是头实体，其余关系里是尾实体
        film, other = (head, tail) if relation.type == "has_genre" else (tail, head)
        bucket = by_film.setdefault(normalize(film), {})
        bucket.setdefault(relation.type, set()).add(normalize(other))

    print(f"图谱：实体 {len(names)}，关系 {len(graph.all_relations())}")
    print(f"参照：影片 {len(reference['films'])} 部（来自数据集 infobox 粗抽取）\n")
    print(f"{'关系':<12}{'参照条数':>9}{'抽出条数':>9}{'命中':>7}{'召回':>9}{'精确率*':>10}")
    print("-" * 60)

    misses: Dict[str, List[str]] = {}
    for relation_type, key, _, precision_meaningful in FIELDS:
        truth_total = 0
        hit_total = 0
        got_total = 0
        sample: List[str] = []
        for film in reference["films"]:
            film_key = normalize(film["title"])
            truth = {normalize(n) for n in film.get(key, []) if normalize(n)}
            got = by_film.get(film_key, {}).get(relation_type, set())
            truth_total += len(truth)
            got_total += len(got)
            hits, matched = match_count(truth, got)
            hit_total += hits
            for missed in sorted(truth - matched):
                if len(sample) < args.samples:
                    sample.append(f"《{film['title']}》 缺 {missed}")
        misses[relation_type] = sample
        recall = hit_total / truth_total if truth_total else 0.0
        precision = hit_total / got_total if got_total else 0.0
        precision_cell = f"{precision:>9.1%}" if precision_meaningful else "      n/a"
        print(
            f"{relation_type:<12}{truth_total:>9}{got_total:>9}{hit_total:>7}"
            f"{recall:>9.1%}{precision_cell}"
        )

    print("\n* 精确率仅对 infobox 能覆盖全集的字段有意义。acted_in 与 produced 的")
    print("  参照只含主演与主要出品方，正文里的配角、联合出品方不在其中，")
    print("  多抽出来的不算错，故不计精确率。")

    print("\n漏抽样例：")
    for relation_type, sample in misses.items():
        if not sample:
            continue
        print(f"  [{relation_type}]")
        for line in sample:
            print(f"    {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
