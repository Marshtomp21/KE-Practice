"""规则抽取器：LLM 不可用或调用失败时的兜底通道。

模式表来自 config/patterns.yaml，代码里不出现任何具体人名、片名或关系字面量。
抽取以句子为单位进行，每句维护一个"语境影片"：本句书名号内的片名，没有则退回
条目自身，因此 "由某人执导" 这类省略宾语的表述也能补全。语境的作用范围由
settings 的 extraction.movie_context_scope 决定——影片条目里跨句沿用容易把
"前作/续作" 的片名带到本片身上，所以默认只在句内生效。

语料若不含影人条目（例如维基影片条目集），人名不会出现在词表里，此时靠规则的
allow_new_head 从谓词上下文里新建实体，再用 looks_like_name 挡住不像名字的捕获。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

from ..core.config import SchemaRegistry, load_schema, load_settings
from ..core.interfaces import TripleExtractor
from ..core.types import Chunk, CleanedDocument, Entity, Evidence, Relation
from .lexicon import TITLE_BRACKET, EntityLexicon

SENTENCE_SPLIT = re.compile(r"[^。！？\n]+[。！？]?")
NAME_SEPARATOR = re.compile(r"[、,，和与及]")
YEAR_IN_TEXT = re.compile(r"(1[89]\d{2}|20\d{2})\s*年")

# 语料里没有影人条目时，人名只能由规则从上下文里新建。以下三组约束用来挡住
# 「由乔治·米勒执导」这类捕获里混进来的虚词与非名称片段。
LEADING_PARTICLES = "由为与和及在被该并同兼等而则后前其他她它这那亦也又更将于自从到至以是把对向且随因当如若但却已再共另均各每"
TRAILING_PARTICLES = "的了着过之及与和并且或者等再也就还都很更最"
# 名称里不该出现的字：出现即判定捕获跨越了短语边界
NAME_FORBIDDEN = set("电影影片本片该片导演编剧演员主演出品发行剧情故事作品系列票房年月日第部集季")
NAME_ALLOWED = re.compile(r"^[一-龥A-Za-z0-9·\.\-—－ ]+$")
# 整词黑名单：这些词形状上像名字（2-4 个汉字），但语义上是职能或动作。
# 用整词而不是单字，避免误伤含同一个字的真实人名。
NAME_BLOCKLIST = {
    "客串", "特别出演", "友情出演", "旁白", "配音", "监制", "制片", "制作",
    "摄影", "剪辑", "配乐", "美术", "造型", "特效", "字幕", "原作", "原著",
    "主唱", "作曲", "作词", "编曲", "翻译", "改编", "策划", "统筹", "宣传",
    "本片", "该片", "影片", "全片", "此片", "续集", "前作", "系列", "官方",
    "同名", "以及", "其中", "包括", "另外", "此外", "目前", "随后", "最终",
}


def looks_like_name(token: str, minimum: int = 2, maximum: int = 14) -> bool:
    """新建实体名的可信度判据：长度合理、字符集干净、不含短语标志字。"""
    text = (token or "").strip()
    if not (minimum <= len(text) <= maximum):
        return False
    if not NAME_ALLOWED.match(text):
        return False
    if any(ch in NAME_FORBIDDEN for ch in text):
        return False
    if text in NAME_BLOCKLIST:
        return False
    # 纯数字或纯标点不是名字
    return any("一" <= ch <= "龥" or ch.isalpha() for ch in text)


PARENTHETICAL = re.compile(r"[（(〔\[][^（()）〔〕\[\]]{0,40}[)）〕\]]")


def name_shape_ok(token: str) -> bool:
    """不在词表里时的兜底判据：这串文字长得像不像一个名字。

    词表来自维基链接，没被链接的人名（正文里直接写出来的）会漏掉，
    所以再给一条形状上的通道，但收得很紧：
    - 含间隔号的音译名，如「大卫·史托顿」；
    - 2 到 4 个汉字的中文姓名，如「王家卫」。
    「雷奇似乎不太可能回归」这类短语两条都不满足，仍会被挡下。
    """
    text = (token or "").strip()
    if not looks_like_name(text):
        return False
    if "·" in text and 3 <= len(text) <= 14:
        return all(ch == "·" or "一" <= ch <= "龥" or ch.isalpha() for ch in text)
    return 2 <= len(text) <= 4 and all("一" <= ch <= "龥" for ch in text)


def strip_particles(token: str) -> str:
    """剥掉捕获两端的虚词与括号补注。

    正则只能锚定谓词，主语的左边界要靠这一步收敛。中文维基惯于在人名后附注
    外文原名（「大卫·史托顿（David Stoten）执导」），不去掉的话捕获永远对不上词表。
    """
    text = PARENTHETICAL.sub("", token or "")
    # 「宣布由班·泰勒」这类捕获里，真正的名字在最后一个「由」或冒号之后
    for marker in ("由", "：", ":"):
        position = text.rfind(marker)
        if 0 <= position < len(text) - 2:
            text = text[position + len(marker) :]
    text = text.strip().strip("《》「」『』（）()，,。、；;：: \t")
    while len(text) > 2 and text[0] in LEADING_PARTICLES:
        text = text[1:]
    while len(text) > 2 and text[-1] in TRAILING_PARTICLES:
        text = text[:-1]
    return text.strip()


def entity_key(name: str, entity_type: str) -> str:
    """图内节点主键。归一化阶段可能把多个 key 并成一个，但格式保持不变。"""
    return f"{entity_type}::{name}"


@dataclass
class PatternRule:
    relation: str
    regex: re.Pattern[str]
    head_source: str
    tail_source: str
    head_types: List[str]
    tail_types: List[str]
    split_head: bool = False
    split_tail: bool = False
    allow_new_head: bool = False
    allow_new_tail: bool = False
    confidence: float = 0.8


def load_rules(path: Optional[Path] = None) -> List[PatternRule]:
    target = Path(path) if path else load_settings().path("paths.schema_file").parent / "patterns.yaml"
    payload = yaml.safe_load(Path(target).read_text(encoding="utf-8")) or {}
    placeholders: Dict[str, str] = payload.get("placeholders", {})
    rules: List[PatternRule] = []
    for item in payload.get("rules", []):
        expression = item["pattern"]
        for token, replacement in placeholders.items():
            expression = expression.replace("{" + token + "}", replacement)
        rules.append(
            PatternRule(
                relation=item["relation"],
                regex=re.compile(expression),
                head_source=item.get("head", "capture"),
                tail_source=item.get("tail", "capture"),
                head_types=list(item.get("head_types", [])),
                tail_types=list(item.get("tail_types", [])),
                split_head=bool(item.get("split_head", False)),
                split_tail=bool(item.get("split_tail", False)),
                allow_new_head=bool(item.get("allow_new_head", False)),
                allow_new_tail=bool(item.get("allow_new_tail", False)),
                confidence=float(item.get("confidence", 0.8)),
            )
        )
    return rules


class RuleExtractor(TripleExtractor):
    """基于模式表 + 实体词表的确定性抽取。"""

    name = "rule"

    def __init__(
        self,
        lexicon: EntityLexicon,
        schema: Optional[SchemaRegistry] = None,
        rules: Optional[Sequence[PatternRule]] = None,
        context_scope: str = "sentence",
        shape_fallback: bool = True,
    ) -> None:
        self.lexicon = lexicon
        self.schema = schema or load_schema()
        self.rules = list(rules) if rules is not None else load_rules()
        self.context_scope = context_scope
        self.shape_fallback = shape_fallback
        self._context_movie: Dict[str, str] = {}

    def extract(
        self, chunk: Chunk, document: CleanedDocument
    ) -> Tuple[List[Entity], List[Relation]]:
        entities: Dict[str, Entity] = {}
        relations: Dict[str, Relation] = {}

        subject_type = self.lexicon.resolve(document.title, [document.entity_type])
        subject = (document.title, subject_type or document.entity_type)

        document_fallback = subject[0] if subject[1] == "Movie" else None
        carried = self._context_movie.get(document.doc_id) or None
        for sentence, sentence_offset in self._sentences(chunk.text):
            movie_here = self._movie_in_sentence(sentence)
            if movie_here:
                carried = movie_here
            if self.context_scope == "document":
                context_movie = carried or document_fallback
            else:
                context_movie = movie_here or document_fallback
            year = self._year_in_sentence(sentence)

            for rule in self.rules:
                for match in rule.regex.finditer(sentence):
                    heads = self._endpoint(
                        rule.head_source, rule.head_types, match, "head",
                        sentence, subject, context_movie,
                        rule.split_head, rule.allow_new_head,
                    )
                    tails = self._endpoint(
                        rule.tail_source, rule.tail_types, match, "tail",
                        sentence, subject, context_movie,
                        rule.split_tail, rule.allow_new_tail,
                    )
                    if not heads or not tails:
                        continue

                    span = (
                        chunk.char_offset + sentence_offset + match.start(),
                        chunk.char_offset + sentence_offset + match.end(),
                    )
                    evidence = Evidence(
                        doc_id=document.doc_id,
                        chunk_id=chunk.id,
                        char_start=span[0],
                        char_end=span[1],
                        raw_text=sentence.strip(),
                        confidence=rule.confidence,
                    )
                    for head_name, head_type in heads:
                        for tail_name, tail_type in tails:
                            if head_name == tail_name:
                                continue
                            self._record(
                                entities, relations, rule, evidence, year,
                                (head_name, head_type), (tail_name, tail_type),
                            )
        self._context_movie[document.doc_id] = carried or ""
        return list(entities.values()), list(relations.values())

    # ---- 内部工具 -------------------------------------------------------

    def _sentences(self, text: str) -> List[Tuple[str, int]]:
        return [(m.group(), m.start()) for m in SENTENCE_SPLIT.finditer(text) if m.group().strip()]

    def _movie_in_sentence(self, sentence: str) -> Optional[str]:
        titles = TITLE_BRACKET.findall(sentence)
        return titles[-1] if titles else None

    def _year_in_sentence(self, sentence: str) -> Optional[int]:
        match = YEAR_IN_TEXT.search(sentence)
        return int(match.group(1)) if match else None

    def _trim_to_lexicon(self, raw: str, expected: Sequence[str]) -> Optional[Tuple[str, str]]:
        """捕获文字可能带前缀（"李四出演孔瀚"），从右往左找最长的词表命中。"""
        token = strip_particles(raw)
        if not token:
            return None
        for start in range(len(token)):
            candidate = token[start:]
            resolved = self.lexicon.resolve(candidate, expected)
            if resolved:
                return candidate, resolved
        return None

    def _trim_to_known(self, token: str) -> Optional[str]:
        """把捕获收敛到已知实体名上。

        正则只能锚定谓词，主语的左边界靠这一步定：从左往右逐字缩短，
        取第一个落在词表里的后缀。词表为空时（例如换了没有链接标注的语料）
        直接放行，退回到纯规则模式。
        """
        known = self.lexicon.known_names
        if not known:
            return token
        if not token:
            return None
        for start in range(len(token) - 1):
            candidate = token[start:]
            if candidate in known:
                return candidate
        return None

    def _endpoint(
        self,
        source: str,
        types: Sequence[str],
        match: re.Match[str],
        group: str,
        sentence: str,
        subject: Tuple[str, str],
        context_movie: Optional[str],
        split: bool,
        allow_new: bool,
    ) -> List[Tuple[str, str]]:
        if source == "subject":
            return [subject] if not types or subject[1] in types else []
        if source == "context_movie":
            return [(context_movie, "Movie")] if context_movie else []
        if source == "subject_or_person":
            for _, _, name in self.lexicon.scan(sentence):
                resolved = self.lexicon.resolve(name, ["Person"])
                if resolved:
                    return [(name, resolved)]
            return [subject] if not types or subject[1] in types else []

        try:
            raw = match.group(group)
        except (IndexError, KeyError):  # 该规则没有声明这个命名组
            return []
        if raw is None:
            return []

        pieces = [p for p in NAME_SEPARATOR.split(raw) if p.strip()] if split else [raw]
        found: List[Tuple[str, str]] = []
        for piece in pieces:
            hit = self._trim_to_lexicon(piece, types)
            if hit:
                found.append(hit)
            elif allow_new and types:
                stripped = strip_particles(piece)
                token = self._trim_to_known(stripped)
                if token and looks_like_name(token):
                    found.append((token, types[0]))
                elif self.shape_fallback and name_shape_ok(stripped):
                    found.append((stripped, types[0]))
        return found

    def _record(
        self,
        entities: Dict[str, Entity],
        relations: Dict[str, Relation],
        rule: PatternRule,
        evidence: Evidence,
        year: Optional[int],
        head: Tuple[str, str],
        tail: Tuple[str, str],
    ) -> None:
        reason = self.schema.validate_triple(head[1], rule.relation, tail[1])
        if reason:
            return  # 由调用方统一记录被拒三元组，这里只负责不产出脏数据

        for name, entity_type in (head, tail):
            key = entity_key(name, entity_type)
            entity = entities.get(key)
            if entity is None:
                entity = Entity(id=key, name=name, type=entity_type)
                entities[key] = entity
            entity.evidences.append(evidence)

        head_key = entity_key(*head)
        tail_key = entity_key(*tail)
        relation_id = f"{head_key}|{rule.relation}|{tail_key}"
        relation = relations.get(relation_id)
        if relation is None:
            relation = Relation(
                id=relation_id,
                head_id=head_key,
                tail_id=tail_key,
                type=rule.relation,
                start_year=year,
                end_year=year,
                attributes={"extractor": self.name},
            )
            relations[relation_id] = relation
        elif relation.start_year is None and year is not None:
            relation.start_year = year
            relation.end_year = year
        relation.evidences.append(evidence)

    def rejected_reason(self, head_type: str, relation: str, tail_type: str) -> Optional[str]:
        return self.schema.validate_triple(head_type, relation, tail_type)
