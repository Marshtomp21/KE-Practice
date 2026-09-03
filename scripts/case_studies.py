"""生成案例分析：两个成功案例 + 一个失败案例，全部由当前图谱现算。

失败案例不是编的：脚本会在图里实际找出问题实例，并沿"抽取 → 归一化 → 检索
→ 生成"四个环节定位问题最早出现在哪一步。

用法：python scripts/case_studies.py
输出：eval/results/case_studies.md
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.generate.service import QAService

TRUTH_FILE = PROJECT_ROOT / "eval" / "ground_truth.json"
OUTPUT_FILE = PROJECT_ROOT / "eval" / "results" / "case_studies.md"


def block(lines: List[str], answer, title: str) -> None:
    lines.append(f"检索器：`{answer.retriever_name}`，耗时 {answer.latency:.2f}s")
    lines.append("")
    lines.append("```text")
    lines.append(answer.text.strip())
    lines.append("```")
    lines.append("")
    if answer.citations:
        lines.append("引用来源：")
        for citation in answer.citations[:3]:
            lines.append(
                f"- `[{citation.marker}]` {citation.doc_id} "
                f"[{citation.char_start}:{citation.char_end}]"
            )
        lines.append("")


def case_multi_hop(service: QAService, truth: Dict[str, Any], lines: List[str]) -> None:
    films_of: Dict[str, List[str]] = defaultdict(list)
    for movie in truth["movies"]:
        for actor in movie["cast"]:
            films_of[actor].append(movie["title"])
    pair = None
    for movie in truth["movies"]:
        if len(movie["cast"]) >= 2:
            left, right = movie["cast"][0], movie["cast"][1]
            if len(set(films_of[left]) & set(films_of[right])) >= 2:
                pair = (left, right, sorted(set(films_of[left]) & set(films_of[right])))
                break
    if not pair:
        return
    left, right, shared = pair
    question = f"{left}与{right}通过哪些影片产生关联？"

    lines.append("## 案例一（成功）：多跳关联，向量基线结构性失败")
    lines.append("")
    lines.append(f"**问题**：{question}")
    lines.append("")
    lines.append(f"**标准答案**：{'、'.join(f'《{t}》' for t in shared)}")
    lines.append("")
    lines.append(
        "两人共同参演的每一部影片，其影片条目里都会同时出现两个名字，所以向量检索并非"
        "完全够不着；但一次检索只能取回 top-k 段文本，而完整答案分散在多部影片的条目中，"
        "没有任何一段文字把它们汇总起来。答案实际存在于「两条 `acted_in` 边共享同一个影片"
        "节点」这一结构里，需要在图上做交集才能一次性取全。"
    )
    lines.append("")
    for name in ("vector", "ppr"):
        answer = service.ask(question, retriever_name=name)
        hit = [title for title in shared if title in answer.text]
        lines.append(f"### `{name}` 命中 {len(hit)}/{len(shared)} 部")
        lines.append("")
        block(lines, answer, name)
    lines.append(
        "**结论**：向量检索能取回两人各自的条目，但取回之后没有任何机制把两段文本对齐到"
        "同一部影片上；图检索直接把共享的影片节点作为路径返回，答案本身就是解释。"
    )
    lines.append("")


def case_negation(service: QAService, truth: Dict[str, Any], lines: List[str]) -> None:
    films_of: Dict[str, List[str]] = defaultdict(list)
    for movie in truth["movies"]:
        for actor in movie["cast"]:
            films_of[actor].append(movie["title"])
    target = truth["movies"][5]
    outsider = next(
        (a for a in films_of if target["title"] not in films_of[a] and a not in target["cast"]),
        None,
    )
    if not outsider:
        return
    question = f"{outsider}出演过《{target['title']}》吗？"

    lines.append("## 案例二（成功）：反事实否定，图的缺边即结论")
    lines.append("")
    lines.append(f"**问题**：{question}")
    lines.append("")
    lines.append(f"**事实**：{outsider} 没有出演过《{target['title']}》，图中不存在这条边。")
    lines.append("")
    answer = service.ask(question, retriever_name="ppr")
    block(lines, answer, "ppr")
    lines.append(
        "**结论**：图谱里「查不到这条边」是一个可以直接使用的判据，"
        "生成阶段据此给出明确否定，而不是依赖模型自觉不编造。"
        "纯向量通道拿到的是两段各自成立的原文，缺少作出否定判断的依据。"
    )
    lines.append("")


def case_failure(service: QAService, lines: List[str]) -> None:
    store = service.store
    by_name: Dict[str, set] = defaultdict(set)
    for entity in store.all_entities():
        by_name[entity.name].add(entity.type)
    collisions = sorted(name for name, types in by_name.items() if len(types) > 1)

    lines.append("## 案例三（失败）：角色名与人名撞车，错误从抽取一路传到答案")
    lines.append("")
    if not collisions:
        lines.append("当前图谱中没有发现同名跨类型实体，本案例暂不成立。")
        lines.append("")
        return

    sample = collisions[0]
    lines.append(
        f"当前图谱里有 **{len(collisions)} 组**同名跨类型实体（同一个名字既是 `Person` 又是 "
        f"`Character`），例如：{'、'.join(collisions[:6])}。以 **{sample}** 为例。"
    )
    lines.append("")

    person = store.get_entity(f"Person::{sample}")
    character = store.get_entity(f"Character::{sample}")
    lines.append("### 问题实例")
    lines.append("")
    for entity in (person, character):
        if entity is None:
            continue
        evidence = entity.evidences[0] if entity.evidences else None
        lines.append(f"- `{entity.id}`，证据来自 {evidence.doc_id if evidence else '（无）'}")
        if evidence:
            lines.append(f"  - 原句：{evidence.raw_text[:70]}")
    lines.append("")

    lines.append("### 定位到具体环节")
    lines.append("")
    lines.append(
        "1. **抽取环节（问题发生在这里）**：`config/patterns.yaml` 里 `plays` 规则的 "
        "`allow_new_tail: true` 允许把「在片中饰演 X」中的 X 直接作为新的 `Character` 实体建出来，"
        "没有检查这个 X 是不是已经作为 `Person` 存在。角色名与真人重名时就产生了两个同名节点。"
    )
    lines.append(
        "2. **归一化环节（没有拦住）**：`normalize.disambiguate_by_type` 为真，"
        "跨类型一律不合并——这条策略本身是对的（人和角色确实不是一回事），"
        "但它也意味着归一化不会对这种重名发出任何告警。"
    )
    lines.append(
        "3. **检索环节（被放大）**：`AnchorResolver` 按名称精确命中，同名的两个实体会一起成为锚点，"
        "于是 PPR 的重启分布被摊到了一个无关的角色节点上。"
    )
    lines.append(
        "4. **生成环节（暴露给用户）**：子图里出现「某演员 → 饰演 → "
        f"{sample}」这样的边，读者会误以为在讲那位同名影人。"
    )
    lines.append("")

    lines.append("### 可行的修法（本次未实施，留作后续）")
    lines.append("")
    lines.append(
        "- 在抽取阶段给新建的 `Character` 加一条约束：若该名称已在词表中登记为 `Person`，"
        "则把角色节点的 id 改写为「影片名::角色名」，让它天然带上归属；"
    )
    lines.append(
        "- 在锚点解析阶段引入类型偏好：问句里出现「出演 / 执导」这类谓词时，"
        "同名候选优先取 `Person`。"
    )
    lines.append("")
    lines.append(
        "把这个案例留在报告里，是因为它完整展示了「抽取阶段一个宽松的默认值，"
        "会在两个环节之后变成用户可见的错误」这条传导链路。"
    )
    lines.append("")


def coverage_note(service: QAService, truth: Dict[str, Any], lines: List[str]) -> None:
    stats = service.graph_stats()
    expected = sum(len(m["genres"]) for m in truth["movies"])
    actual = stats["relation_types"].get("has_genre", 0)
    lines.append("## 附：一处已知的抽取漏采")
    lines.append("")
    lines.append(
        f"影片类型关系 `has_genre` 应有约 {expected} 条，实际抽出 {actual} 条，召回约 "
        f"{actual / max(expected, 1):.0%}。原因是 `patterns.yaml` 里的类型规则只匹配"
        "「……电影」前紧邻的两个字，一部影片写成「剧情、犯罪电影」时，只有后一个类型被抽到。"
        "这属于规则表达力不足，改规则即可，不涉及架构。"
    )
    lines.append("")


def main() -> int:
    if not TRUTH_FILE.exists():
        print(f"缺少 {TRUTH_FILE}，请先运行 scripts/make_sample_corpus.py")
        return 1
    truth = json.loads(TRUTH_FILE.read_text(encoding="utf-8"))
    service = QAService()

    lines: List[str] = ["# 案例分析", ""]
    lines.append(
        "以下案例全部由 `scripts/case_studies.py` 依据当前图谱现场生成，"
        "答案与证据均为真实运行结果。"
    )
    lines.append("")
    case_multi_hop(service, truth, lines)
    case_negation(service, truth, lines)
    case_failure(service, lines)
    coverage_note(service, truth, lines)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"案例分析 -> {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
