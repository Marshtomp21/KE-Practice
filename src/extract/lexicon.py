"""实体词表。

语料里每个条目本身就是一个实体（条目标题 + entity_type），这批标题构成了
最可靠的词表；再补上正文里书名号包裹的片名、以及从奖项/类型表述里回收的短语。
规则抽取与检索锚点识别都以这张表为准，避免在正则里硬编码具体人名片名。
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

TITLE_BRACKET = re.compile(r"《([^《》]{1,30})》")


@dataclass
class LexiconEntry:
    name: str
    type: str
    sources: Set[str] = field(default_factory=set)

    @property
    def key(self) -> str:
        return f"{self.type}:{self.name}"


class EntityLexicon:
    """名称 -> 候选类型 的多重映射，支持同名跨类型共存。"""

    def __init__(self) -> None:
        self._by_name: Dict[str, Dict[str, LexiconEntry]] = defaultdict(dict)
        self._known_names: Set[str] = set()
        self._pattern: Optional[re.Pattern[str]] = None

    def add(self, name: str, entity_type: str, source: str = "") -> None:
        token = (name or "").strip()
        if not token or not entity_type:
            return
        bucket = self._by_name[token]
        entry = bucket.get(entity_type)
        if entry is None:
            entry = LexiconEntry(name=token, type=entity_type)
            bucket[entity_type] = entry
        if source:
            entry.sources.add(source)
        self._pattern = None

    def types_of(self, name: str) -> List[str]:
        return list(self._by_name.get((name or "").strip(), {}))

    def has(self, name: str) -> bool:
        return (name or "").strip() in self._by_name

    def resolve(self, name: str, expected: Iterable[str] = ()) -> Optional[str]:
        """在期望的类型集合里挑一个类型；期望为空时返回唯一类型，歧义则返回 None。"""
        candidates = self.types_of(name)
        if not candidates:
            return None
        expected = list(expected)
        if expected:
            narrowed = [t for t in candidates if t in expected]
            if len(narrowed) == 1:
                return narrowed[0]
            return narrowed[0] if narrowed else None
        return candidates[0] if len(candidates) == 1 else None

    def names(self) -> List[str]:
        return list(self._by_name)

    def entries(self) -> List[LexiconEntry]:
        return [entry for bucket in self._by_name.values() for entry in bucket.values()]

    def scan(self, text: str) -> List[Tuple[int, int, str]]:
        """在文本里找出所有词表命中，长词优先，返回 (起, 止, 名称)。"""
        if self._pattern is None:
            ordered = sorted(self._by_name, key=len, reverse=True)
            if not ordered:
                return []
            self._pattern = re.compile("|".join(re.escape(n) for n in ordered))
        hits: List[Tuple[int, int, str]] = []
        for match in self._pattern.finditer(text):
            hits.append((match.start(), match.end(), match.group()))
        return hits

    def load_known_names(self, path: Optional[str]) -> int:
        """载入一份「已知实体名」清单（每行一个），只用于校验新建实体是否可信。

        维基正文里的 [[条目]] 链接就是一份人工标注的实体表，导入阶段把链接目标
        收集成这份清单。规则抽取新建实体时要求候选名在表内，比堆停用词可靠得多。
        表里不带类型：类型由命中的规则决定（"由X执导" 里的 X 必然是 Person）。
        """
        if not path:
            return 0
        target = Path(path)
        if not target.exists():
            return 0
        for line in target.read_text(encoding="utf-8").splitlines():
            token = line.strip()
            if token and len(token) >= 2:
                self._known_names.add(token)
        return len(self._known_names)

    @property
    def known_names(self) -> Set[str]:
        return self._known_names

    @classmethod
    def from_documents(cls, documents: Iterable, movie_type: str = "Movie") -> "EntityLexicon":
        """条目标题按其 entity_type 入表，正文书名号内容按影片入表。"""
        lexicon = cls()
        for document in documents:
            lexicon.add(document.title, document.entity_type, source=document.doc_id)
            for match in TITLE_BRACKET.finditer(document.text):
                lexicon.add(match.group(1), movie_type, source=document.doc_id)
        return lexicon
