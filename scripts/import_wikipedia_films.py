"""把中文维基影片、演员和导演数据集导入 data/raw。

输入是 KE-Practice-main/data/wikipedia_300_films_final/ 下的 films.jsonl：
每条记录含条目导语（intro）、完整 wikitext，以及数据集自带的一份粗抽取结果。

本脚本只负责**数据准备**：把 wikitext 还原成自然语言正文段落，写成本项目统一的
raw JSON。下游的清洗、切分与索引构建由各自模块负责。

为什么正文要在这里处理而不是丢给 TextCleaner：
wikitext 的模板、表格、脚注是**结构**噪声，去掉它们会大幅改变文本长度与内容；
而 TextCleaner 的职责是逐字符清洗并维护偏移映射，需要输入已经是「正文」。
所以这里产出的 text 就是条目正文，raw JSON 里的偏移即正文偏移，证据可反查。

章节取舍：只保留叙事性章节（剧情、演员、制作、发行、评价……），
丢弃参考资料、外部链接、延伸阅读等纯引用章节——它们没有可抽取的语义。

用法：
  python scripts/import_wikipedia_films.py --clean
  python scripts/import_wikipedia_films.py --limit 50      # 先小批量试跑
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest.cleaner import load_variant_table

DEFAULT_SOURCE = PROJECT_ROOT / "data" / "source" / "wikipedia_300_films_final"

# 纯引用性章节，没有可抽取的叙事内容
# 注意：折叠是逐字进行的，「外部連結」折出来是「外部连结」而不是词组级的
# 「外部链接」，所以两种写法都要列上。
SKIP_SECTIONS = {
    "参考资料", "参考文献", "参考来源", "资料来源", "注释", "脚注", "注脚",
    "外部链接", "外部连结", "外部链结", "延伸阅读", "参见", "另见", "参考",
    "相关条目", "相关链接", "画廊", "图库", "来源", "备注", "注释与参考",
    "参考资料及注释", "参考文献与注释", "扩展阅读", "外部連接", "脚注与参考",
}
# 章节标题同样繁简混排，比对前先用清洗层那张对照表折成简体，
# 复用同一张表可以避免两处规则漂移
_VARIANTS = load_variant_table(str(PROJECT_ROOT / "config" / "zh_variants.txt"))


def fold_variants(text: str) -> str:
    return "".join(_VARIANTS.get(ch, ch) for ch in text or "")


COMMENT = re.compile(r"<!--.*?-->", re.S)
REF_PAIR = re.compile(r"<ref[^>/]*>.*?</ref>", re.S | re.I)
REF_SELF = re.compile(r"<ref[^>]*/>", re.I)
HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*={2,6}\s*$", re.M)
FILE_LINK = re.compile(r"\[\[(?:File|Image|檔案|文件|图像|圖像):[^\[\]]*(?:\[\[[^\]]*\]\][^\[\]]*)*\]\]", re.I)
PIPED_LINK = re.compile(r"\[\[([^\[\]|]+)\|([^\[\]|]*)\]\]")
PLAIN_LINK = re.compile(r"\[\[([^\[\]|]+)\]\]")
EXTERNAL_LABELLED = re.compile(r"\[(?:https?:)?//[^\s\]]+\s+([^\]]+)\]")
EXTERNAL_BARE = re.compile(r"\[(?:https?:)?//[^\s\]]+\]")
BOLD_ITALIC = re.compile(r"'{2,5}")
LIST_MARK = re.compile(r"^[*#:;]+\s*", re.M)
CATEGORY_LINE = re.compile(r"^(?:Category|分类|分類)\s*[:：]", re.I)
BLANKS = re.compile(r"\n{2,}")

# 少数模板携带正文信息，先转成纯文本再统一清模板
LINK_TEMPLATE = re.compile(r"\{\{\s*(?:link-[a-z]{2}|tsl|le)\s*\|([^{}|]*)\|?([^{}|]*)[^{}]*\}\}", re.I)

# 维基的字词转换标记 -{zh-cn:甲;zh-tw:乙}-，不处理就会把整串控制字符留在正文里
CONVERT_MARKUP = re.compile(r"-\{(.*?)\}-", re.S)
CONVERT_PREFERENCE = ("zh-cn", "zh-hans", "zh-sg", "zh-my", "zh", "zh-hk", "zh-tw", "zh-hant")
# 模板被清掉后常留下空的成对括号
EMPTY_BRACKETS = re.compile(r"[（(]\s*[)）]|「\s*」|《\s*》|\[\s*\]")


def resolve_conversion(body: str) -> str:
    """从 -{...}- 里挑一个地区变体，优先简体；没有分号语法就原样返回内容。"""
    payload = body.split("|")[-1]
    if ":" not in payload and "：" not in payload:
        return payload.strip()
    options: Dict[str, str] = {}
    for piece in re.split(r"[;；]", payload):
        if ":" not in piece and "：" not in piece:
            continue
        parts = re.split(r"[:：]", piece, maxsplit=1)
        if len(parts) != 2:
            continue
        options[parts[0].strip().lower()] = parts[1].strip()
    for tag in CONVERT_PREFERENCE:
        if options.get(tag):
            return options[tag]
    return next(iter(options.values()), "") if options else payload.strip()


# 维基正文里的 [[条目]] 链接本身就是一份人工标注的实体表：能被链接出去的
# 几乎都是真实存在的人、机构或作品。把链接目标收集起来当作词表，规则抽取
# 新建实体时用它来卡边界，比堆砌停用词可靠得多。
NAMESPACE_LINK = re.compile(
    r"^(?:Category|分类|分類|File|Image|檔案|文件|图像|圖像|Template|模板|"
    r"Help|Wikipedia|WP|Special|Portal|Module|s|wikt|zh|en|ja):",
    re.I,
)
GAZETTEER_TOKEN = re.compile(r"^[一-龥A-Za-z][一-龥A-Za-z0-9·\.\-' ]{1,29}$")


def harvest_link_targets(wikitext: str) -> List[str]:
    """收集 [[目标]] / [[目标|显示]] 中的目标名，剔除命名空间链接与消歧后缀。"""
    found: List[str] = []
    for match in re.finditer(r"\[\[([^\[\]|]+)(?:\|[^\[\]]*)?\]\]", wikitext or ""):
        target = match.group(1).strip()
        if not target or NAMESPACE_LINK.match(target):
            continue
        target = target.split("#")[0].strip()
        # 「猎人克莱文 (电影)」这类消歧后缀在正文里不会出现，去掉后才对得上
        target = re.sub(r"\s*[（(][^（()）]{1,20}[)）]\s*$", "", target).strip()
        folded = fold_variants(target)
        if GAZETTEER_TOKEN.match(folded):
            found.append(folded)
    return found


def strip_balanced(text: str, opener: str, closer: str) -> str:
    """按配对深度删除模板/表格，正则做不到嵌套匹配。"""
    out: List[str] = []
    depth = 0
    index = 0
    length = len(text)
    while index < length:
        if text.startswith(opener, index):
            depth += 1
            index += len(opener)
            continue
        if text.startswith(closer, index) and depth:
            depth -= 1
            index += len(closer)
            continue
        if depth == 0:
            out.append(text[index])
        index += 1
    return "".join(out)


def wikitext_to_prose(wikitext: str) -> str:
    """wikitext -> 正文段落。顺序要紧：先去引用，再解链接，最后清模板与表格。"""
    text = wikitext or ""
    text = COMMENT.sub("", text)
    text = REF_PAIR.sub("", text)
    text = REF_SELF.sub("", text)
    text = CONVERT_MARKUP.sub(lambda m: resolve_conversion(m.group(1)), text)
    text = FILE_LINK.sub("", text)

    # link-xx 这类模板里第一个参数是可读的中文名，保留下来
    text = LINK_TEMPLATE.sub(lambda m: (m.group(1) or m.group(2) or "").strip(), text)
    text = strip_balanced(text, "{|", "|}")     # 表格
    text = strip_balanced(text, "{{", "}}")     # 模板

    text = PIPED_LINK.sub(lambda m: (m.group(2) or m.group(1)).strip(), text)
    text = PLAIN_LINK.sub(lambda m: m.group(1).strip(), text)
    text = EXTERNAL_LABELLED.sub(lambda m: m.group(1).strip(), text)
    text = EXTERNAL_BARE.sub("", text)
    text = HTML_TAG.sub("", text)
    text = BOLD_ITALIC.sub("", text)
    # 模板清掉后剩下的空括号会干扰后续的实体切分
    for _ in range(3):
        replaced = EMPTY_BRACKETS.sub("", text)
        if replaced == text:
            break
        text = replaced
    return text


def split_sections(prose: str) -> List[Tuple[str, str]]:
    """按标题切成 (章节名, 正文) 列表；导语的章节名为空串。"""
    marks = [(m.start(), m.end(), m.group(2)) for m in HEADING.finditer(prose)]
    if not marks:
        return [("", prose)]
    sections: List[Tuple[str, str]] = [("", prose[: marks[0][0]])]
    for index, (_, end, title) in enumerate(marks):
        stop = marks[index + 1][0] if index + 1 < len(marks) else len(prose)
        sections.append((title, prose[end:stop]))
    return sections


def tidy_block(block: str) -> str:
    """清掉列表符号与表格残留竖线，压缩空行。"""
    lines: List[str] = []
    for raw in block.splitlines():
        line = LIST_MARK.sub("", raw).strip()
        if not line or line.startswith("|") or line.startswith("!"):
            continue
        # [[Category:xxx]] 解链后会变成裸的 Category 行，它是分类标记不是正文
        if CATEGORY_LINE.match(line):
            continue
        if len(line) < 2:
            continue
        lines.append(line)
    return BLANKS.sub("\n", "\n".join(lines)).strip()


def source_to_text(source: Dict[str, Any]) -> str:
    """把影片或人物的维基 source_document 统一转换成可检索正文。"""
    parts: List[str] = []
    intro = tidy_block(wikitext_to_prose(source.get("intro") or ""))
    if intro:
        parts.append(intro)

    for name, block in split_sections(wikitext_to_prose(source.get("raw_wikitext") or "")):
        folded = fold_variants(name).strip()
        if not name:
            continue
        if folded in SKIP_SECTIONS:
            continue
        body = tidy_block(block)
        if len(body) >= 30:
            parts.append(f"{folded}\n{body}")
    return "\n\n".join(parts).strip()


def build_document(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    film = record.get("film") or {}
    source = record.get("source_document") or {}
    title = (film.get("name") or "").strip()
    film_id = (film.get("id") or "").strip()
    if not title or not film_id:
        return None

    text = source_to_text(source)
    if len(text) < 80:
        return None

    return {
        "file_name": f"film_{hashlib.sha1(film_id.encode('utf-8')).hexdigest()[:16]}.json",
        "doc_id": f"film_{film_id}",
        "title": title,
        "url": source.get("url", ""),
        "entity_type": "Movie",
        "text": text,
        "infobox": {
            "年份分类": film.get("year_category", ""),
            "来源": source.get("source", ""),
            "许可": source.get("license", ""),
            "修订版本": str(source.get("revision_id", "")),
        },
    }


def build_person_document(record: Dict[str, Any], role: str) -> Optional[Dict[str, Any]]:
    """把演员/导演记录转换成正文；无独立词条时只陈述数据集给出的片单。"""
    person = record.get("person") or {}
    name = fold_variants(wikitext_to_prose(str(person.get("name") or "")).strip())
    person_id = str(person.get("id") or "").strip()
    if not name or not person_id:
        return None

    source = record.get("source_document") or {}
    text = source_to_text(source) if source else ""
    filmography = [
        fold_variants(str(item.get("name") or "").strip())
        for item in record.get("filmography") or []
        if str(item.get("name") or "").strip()
    ]
    if filmography:
        action = "参演或关联的影片" if role == "Actor" else "执导或关联的影片"
        summary = f"{name}是数据集中的{'演员' if role == 'Actor' else '导演'}。{action}包括：" + \
            "、".join(f"《{title}》" for title in filmography) + "。"
        text = f"{text}\n\n{summary}".strip() if text else summary
    if len(text) < 20:
        return None

    digest = hashlib.sha1(f"{role}:{person_id}".encode("utf-8")).hexdigest()[:16]
    return {
        "file_name": f"person_{role.lower()}_{digest}.json",
        "doc_id": f"person_{role.lower()}_{person_id}",
        "title": name,
        "url": source.get("url", ""),
        "entity_type": "Person",
        "text": text,
        "infobox": {
            "角色": "演员" if role == "Actor" else "导演",
            "人物ID": person_id,
            "关联影片数": int(record.get("related_film_count") or len(filmography)),
            "有独立维基正文": bool(record.get("has_wikipedia_text")),
        },
    }


NOISE_NAME = re.compile(r"[{}\[\]|=<>]|^\s*$")



def load_jsonl(path: Path, limit: int = 0) -> Tuple[List[Dict[str, Any]], int]:
    records: List[Dict[str, Any]] = []
    invalid = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                invalid += 1
            if limit and len(records) >= limit:
                break
    return records, invalid


def main() -> int:
    parser = argparse.ArgumentParser(description="导入中文维基影片、演员与导演语料")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="数据集目录")
    parser.add_argument("--out", default="data/raw", help="输出目录")
    parser.add_argument("--limit", type=int, default=0, help="只导入前 N 部，用于试跑")
    parser.add_argument("--films-only", action="store_true", help="不导入演员和导演文档")
    parser.add_argument("--clean", action="store_true", help="导入前清空输出目录")
    args = parser.parse_args()

    films_file = Path(args.source) / "films.jsonl"
    if not films_file.exists():
        print(f"找不到 {films_file}")
        return 1

    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for stale in out_dir.iterdir():
            if stale.is_file():
                stale.unlink()

    records, invalid = load_jsonl(films_file, args.limit)

    written = 0
    skipped = 0
    lengths: List[int] = []
    gazetteer: Counter = Counter()
    for record in records:
        source_doc = record.get("source_document") or {}
        for token in harvest_link_targets(source_doc.get("raw_wikitext") or ""):
            gazetteer[token] += 1
        film_name = fold_variants((record.get("film") or {}).get("name") or "")
        if GAZETTEER_TOKEN.match(film_name):
            gazetteer[film_name] += 1
        document = build_document(record)
        if document is None:
            skipped += 1
            continue
        target_name = document.pop("file_name")
        (out_dir / target_name).write_text(
            json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        lengths.append(len(document["text"]))
        written += 1

    people_written: Counter = Counter()
    people_skipped: Counter = Counter()
    if not args.films_only:
        for file_name, role in (("actors.jsonl", "Actor"), ("directors.jsonl", "Director")):
            source_file = Path(args.source) / file_name
            if not source_file.exists():
                print(f"警告：缺少 {source_file}，跳过该类人物")
                continue
            person_records, person_invalid = load_jsonl(source_file, args.limit)
            invalid += person_invalid
            for record in person_records:
                person = record.get("person") or {}
                person_name = fold_variants(wikitext_to_prose(str(person.get("name") or "")))
                if GAZETTEER_TOKEN.match(person_name):
                    gazetteer[person_name] += 1
                source_doc = record.get("source_document") or {}
                for token in harvest_link_targets(source_doc.get("raw_wikitext") or ""):
                    gazetteer[token] += 1
                document = build_person_document(record, role)
                if document is None:
                    people_skipped[role] += 1
                    continue
                target_name = document.pop("file_name")
                (out_dir / target_name).write_text(
                    json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                lengths.append(len(document["text"]))
                people_written[role] += 1

    gazetteer_file = PROJECT_ROOT / "data" / "interim" / "wiki_gazetteer.txt"
    gazetteer_file.parent.mkdir(parents=True, exist_ok=True)
    # 只出现一次的链接目标多半是长尾条目，留着也无妨；这里按出现次数排序便于人工查看
    gazetteer_file.write_text(
        "\n".join(name for name, _ in gazetteer.most_common()) + "\n", encoding="utf-8"
    )


    lengths.sort()
    median = lengths[len(lengths) // 2] if lengths else 0
    print(f"导入影片 {written} 篇，跳过 {skipped} 篇")
    if not args.films_only:
        print(
            f"导入演员 {people_written['Actor']} 篇、导演 {people_written['Director']} 篇；"
            f"跳过人物 {sum(people_skipped.values())} 篇"
        )
    print(f"JSON 解析失败 {invalid} 行；输出 -> {out_dir}")
    print(f"正文长度：中位 {median} 字，最短 {lengths[0] if lengths else 0}，最长 {lengths[-1] if lengths else 0}")
    print(f"链接词表 {len(gazetteer)} 个名称 -> {gazetteer_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
