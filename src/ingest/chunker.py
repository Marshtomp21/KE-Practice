"""语义切分：先按段落，再按句读，绝不按固定字符数硬切。

流程：
1. 以换行切出自然段；
2. 逐段累加，累加长度接近 target 时收束成一个片段；
3. 单段本身超过 max 时，用中文句末标点做二次切分，仍然超长才在逗号处让步；
4. 过短的尾片段并回上一片段，避免产出无信息量的碎片。
每个片段都记录它在清洗后正文中的起始偏移。
"""
from __future__ import annotations

import re
from typing import List, Tuple

from ..core.types import Chunk, CleanedDocument

SENTENCE_END = "。！？!?；;"
SOFT_BREAK = "，,、）)】」』"


def _split_keep_offset(
    text: str, base: int, breakers: str, limit: int
) -> List[Tuple[int, str]]:
    """在 breakers 内的字符之后断开；累积长度到 limit 才收一段。

    返回 (相对清洗后正文的偏移, 文本) 列表。
    """
    pieces: List[Tuple[int, str]] = []
    buffer: List[str] = []
    buffer_start = base
    for index, ch in enumerate(text):
        buffer.append(ch)
        if ch in breakers and len(buffer) >= limit:
            pieces.append((buffer_start, "".join(buffer)))
            buffer = []
            buffer_start = base + index + 1
    if buffer:
        pieces.append((buffer_start, "".join(buffer)))
    return pieces


class SemanticChunker:
    """按段落与句读切分，长度参数全部来自 settings.yaml。"""

    def __init__(
        self, target_chars: int = 400, max_chars: int = 700, min_chars: int = 60
    ) -> None:
        self.target_chars = target_chars
        self.max_chars = max_chars
        self.min_chars = min_chars

    def split(self, document: CleanedDocument) -> List[Chunk]:
        paragraphs = [
            (match.start(), match.group())
            for match in re.finditer(r"[^\n]+", document.text)
        ]

        units: List[Tuple[int, str]] = []
        for offset, paragraph in paragraphs:
            if len(paragraph) <= self.max_chars:
                units.append((offset, paragraph))
                continue
            sentences = _split_keep_offset(
                paragraph, offset, SENTENCE_END, self.target_chars
            )
            for sent_offset, sentence in sentences:
                if len(sentence) <= self.max_chars:
                    units.append((sent_offset, sentence))
                else:
                    units.extend(
                        _split_keep_offset(
                            sentence, sent_offset, SOFT_BREAK, self.target_chars
                        )
                    )

        merged = self._merge_short_units(units)
        return [
            Chunk(
                id=f"{document.doc_id}::c{seq:04d}",
                doc_id=document.doc_id,
                text=text,
                char_offset=offset,
                metadata={
                    "title": document.title,
                    "url": document.url,
                    "entity_type": document.entity_type,
                    "sequence": seq,
                    "raw_span": list(
                        document.to_raw_span(offset, offset + len(text))
                    ),
                },
            )
            for seq, (offset, text) in enumerate(merged)
        ]

    def _merge_short_units(
        self, units: List[Tuple[int, str]]
    ) -> List[Tuple[int, str]]:
        """把相邻的短单元黏回去，直到接近 target 长度为止。"""
        merged: List[Tuple[int, str]] = []
        for offset, text in units:
            if not merged:
                merged.append((offset, text))
                continue
            prev_offset, prev_text = merged[-1]
            gap = offset - (prev_offset + len(prev_text))
            joinable = 0 <= gap <= 1
            too_short = len(prev_text) < self.min_chars or len(text) < self.min_chars
            fits = len(prev_text) + gap + len(text) <= self.max_chars
            under_target = len(prev_text) < self.target_chars
            if joinable and fits and (too_short or under_target):
                merged[-1] = (prev_offset, prev_text + " " * gap + text)
            else:
                merged.append((offset, text))
        return [(offset, text) for offset, text in merged if text.strip()]
