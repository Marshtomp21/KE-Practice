"""正文清洗，并同步维护字符偏移映射。

清洗逐字符进行：每保留一个字符，就把它在原文中的下标压进 offset_map。
因此 offset_map[i] 即清洗后第 i 个字符的原文位置，证据区间随时可反查原文。

中文维基的条目繁简混排（同一批语料里「執導」出现 336 次、「执导」170 次），
不统一会让「劉德華」与「刘德华」变成图上两个节点，抽取规则也要写两套。
因此清洗阶段做一次繁到简的折叠，对照表来自 config/zh_variants.txt，
**只收一对一的单字映射**：替换不改变文本长度，偏移映射与证据区间保持精确。
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..core.types import CleanedDocument, RawDocument

# 全角标点在中文正文里保留原样，只把全角字母数字与空格折成半角
_FULLWIDTH_START = 0xFF01
_FULLWIDTH_END = 0xFF5E
_FULLWIDTH_SHIFT = 0xFEE0
_IDEOGRAPHIC_SPACE = 0x3000


def _fold_char(ch: str) -> str:
    code = ord(ch)
    if code == _IDEOGRAPHIC_SPACE:
        return " "
    if _FULLWIDTH_START <= code <= _FULLWIDTH_END:
        folded = chr(code - _FULLWIDTH_SHIFT)
        if folded.isalnum():
            return folded
    return ch


@lru_cache(maxsize=2)
def load_variant_table(path: Optional[str] = None) -> Dict[str, str]:
    """读取繁简单字对照表。文件缺失时返回空表，清洗照常进行，只是不折叠。"""
    if path is None:
        return {}
    target = Path(path)
    if not target.exists():
        return {}
    table: Dict[str, str] = {}
    for line in target.read_text(encoding="utf-8").splitlines():
        row = line.strip()
        if not row or row.startswith("#") or len(row) != 2:
            continue
        table[row[0]] = row[1]
    return table


def _masked_spans(text: str, patterns: Sequence[str]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for pattern in patterns:
        try:
            compiled = re.compile(pattern)
        except re.error:
            continue
        spans.extend((m.start(), m.end()) for m in compiled.finditer(text))
    return spans


class TextCleaner:
    """把原始正文洗成可抽取的干净文本，同时产出偏移映射。"""

    def __init__(
        self, drop_patterns: Iterable[str] = (), variant_table_path: Optional[str] = None
    ) -> None:
        self.drop_patterns = list(drop_patterns)
        self.variants = load_variant_table(variant_table_path)

    def clean(self, document: RawDocument) -> CleanedDocument:
        source = document.text.replace("\r\n", "\n").replace("\r", "\n")
        dropped = _masked_spans(source, self.drop_patterns)
        drop_flags = bytearray(len(source))
        for start, end in dropped:
            for i in range(start, end):
                drop_flags[i] = 1

        kept_chars: List[str] = []
        offset_map: List[int] = []
        pending_blank = False       # 待压缩的空白
        pending_newline = False     # 待压缩的换行（优先级高于空格）

        for index, ch in enumerate(source):
            if drop_flags[index]:
                continue
            if ch == "\n":
                pending_newline = True
                continue
            if ch.isspace() or ord(ch) == _IDEOGRAPHIC_SPACE:
                pending_blank = True
                continue
            if unicodedata.category(ch) in ("Cc", "Cf"):
                continue

            if kept_chars and (pending_newline or pending_blank):
                separator = "\n" if pending_newline else " "
                kept_chars.append(separator)
                offset_map.append(index)
            pending_newline = False
            pending_blank = False

            folded = _fold_char(ch)
            # 繁简折叠放在最后一步，且严格一字换一字，长度不变
            kept_chars.append(self.variants.get(folded, folded))
            offset_map.append(index)

        return CleanedDocument(
            doc_id=document.doc_id,
            title=document.title,
            url=document.url,
            entity_type=document.entity_type,
            text="".join(kept_chars),
            offset_map=offset_map,
            infobox=document.infobox,
        )
