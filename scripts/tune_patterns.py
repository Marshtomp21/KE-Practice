"""规则调试台：逐条展示每个模式在真实语料上抽到了什么。

写规则最容易犯的错是凭想象写、不看语料。这个脚本把每条规则的命中样例、
产出数量、以及被 looks_like_name 挡掉的比例打出来，改一轮看一轮。

用法：
  python scripts/tune_patterns.py                 # 全部规则概览
  python scripts/tune_patterns.py --rule directed # 只看某个关系，附命中原句
  python scripts/tune_patterns.py --docs 80       # 只跑前 80 篇，快速迭代
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import load_schema, load_settings
from src.extract.lexicon import EntityLexicon
from src.extract.rule_extractor import RuleExtractor, looks_like_name, strip_particles
from src.ingest.pipeline import iter_persisted_documents, load_persisted_chunks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule", default="", help="只看这个关系类型")
    parser.add_argument("--docs", type=int, default=0, help="只跑前 N 篇文档")
    parser.add_argument("--samples", type=int, default=12, help="每条规则展示几个样例")
    args = parser.parse_args()

    settings = load_settings()
    documents = list(iter_persisted_documents(settings))
    if args.docs:
        documents = documents[: args.docs]
    keep = {d.doc_id for d in documents}
    chunks = [c for c in load_persisted_chunks(settings) if c.doc_id in keep]
    if not chunks:
        print("没有切分结果，请先运行 python scripts/build_index.py")
        return 1

    lexicon = EntityLexicon.from_documents(documents)
    extractor = RuleExtractor(
        lexicon,
        schema=load_schema(),
        context_scope=str(settings.get("extraction.movie_context_scope", "sentence")),
    )
    by_doc = {d.doc_id: d for d in documents}

    print(f"语料：文档 {len(documents)}，片段 {len(chunks)}，规则 {len(extractor.rules)} 条\n")

    produced: Counter = Counter()
    samples: Dict[str, List[str]] = {}
    rejected: Counter = Counter()

    for index, rule in enumerate(extractor.rules):
        if args.rule and rule.relation != args.rule:
            continue
        tag = f"{rule.relation}#{index}"
        for chunk in chunks:
            document = by_doc[chunk.doc_id]
            subject_type = lexicon.resolve(document.title, [document.entity_type])
            subject = (document.title, subject_type or document.entity_type)
            for sentence, _ in extractor._sentences(chunk.text):
                movie_here = extractor._movie_in_sentence(sentence)
                context = movie_here or (subject[0] if subject[1] == "Movie" else None)
                for match in rule.regex.finditer(sentence):
                    heads = extractor._endpoint(
                        rule.head_source, rule.head_types, match, "head",
                        sentence, subject, context, rule.split_head, rule.allow_new_head,
                    )
                    tails = extractor._endpoint(
                        rule.tail_source, rule.tail_types, match, "tail",
                        sentence, subject, context, rule.split_tail, rule.allow_new_tail,
                    )
                    if not heads or not tails:
                        # 记一下是被名称判据挡掉的，还是压根没捕获到
                        try:
                            raw = match.group("head")
                        except (IndexError, KeyError):
                            raw = ""
                        if raw and not looks_like_name(strip_particles(raw)):
                            rejected[tag] += 1
                        continue
                    produced[tag] += len(heads) * len(tails)
                    if len(samples.setdefault(tag, [])) < args.samples:
                        pair = f"{heads[0][0]} -> {tails[0][0]}"
                        samples[tag].append(f"{pair:<40} ⟵ {match.group()[:60]}")

        status = f"{tag:<22} 产出 {produced[tag]:>5}  被名称判据挡掉 {rejected[tag]:>5}"
        print(status)
        print(f"    模式 {rule.regex.pattern[:96]}")
        for line in samples.get(tag, [])[: args.samples]:
            print(f"      {line}")
        if not samples.get(tag):
            print("      （无命中）")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
