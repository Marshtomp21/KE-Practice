"""Conservative, query-local relation verification with exact source spans.

This is a movie-domain verifier, not an offline knowledge extraction pipeline.
Mere co-occurrence and model-generated facts are never accepted as edges.
"""
from __future__ import annotations

import hashlib
import re

from ..core.types import Chunk, Entity, Evidence, Relation


class EvidenceVerifier:
    def __init__(self, entities: dict[str, Entity]) -> None:
        self.entities = entities
        self.surfaces: dict[str, set[str]] = {}
        for entity in entities.values():
            for surface in entity.surface_forms():
                if len(surface) >= 2:
                    self.surfaces.setdefault(surface, set()).add(entity.id)
        self.pattern = re.compile("|".join(re.escape(s) for s in sorted(self.surfaces, key=len, reverse=True)) or r"(?!)")

    def mentions(self, text: str, kind: str) -> set[str]:
        return {entity_id for match in self.pattern.finditer(text)
                for entity_id in self.surfaces[match.group()] if self.entities[entity_id].type == kind
                and len(self.surfaces[match.group()]) == 1}

    def verify(self, chunk: Chunk) -> list[Relation]:
        # Document identity supplies context, never relation truth.
        movie_ids = self.mentions(str(chunk.metadata.get("title", "")), "Movie")
        doc_entity = chunk.doc_id.removeprefix("film_")
        if chunk.doc_id.startswith("film_") and doc_entity in self.entities:
            if self.entities[doc_entity].type == "Movie":
                movie_ids = {doc_entity}
        result: dict[tuple[str, str, str], Relation] = {}
        person_id = re.sub(r"^person_(?:actor|director)_", "", chunk.doc_id)
        if not chunk.doc_id.startswith(("person_actor_", "person_director_")) or person_id not in self.entities:
            person_id = None

        def add(person, role, movie, start, end):
            span = Evidence(chunk.doc_id, chunk.id, chunk.char_offset + start,
                            chunk.char_offset + end, chunk.text[start:end], 0.95)
            key = (person, role, movie)
            token = "|".join((*key, chunk.id, str(start)))
            result[key] = Relation("temporary-" + hashlib.sha1(token.encode()).hexdigest(),
                                   person, movie, role, attributes={"temporary": True,
                                   "verifier": "explicit-role-span"}, evidences=[span])
        for match in re.finditer(r"[^。！？\n]+[。！？]?", chunk.text):
            sentence = match.group()
            if "或关联" in sentence:
                continue
            # Conservative rejection includes hypothetical, negated and cancelled casting.
            if re.search(r"未曾|没有|并未|并非|未参演|未出演|不曾|否认|传闻|原定|原计划|拟邀|有望|辞演|退出|取代|拒绝", sentence):
                continue
            explicit_movies = self.mentions(sentence, "Movie")
            raw_titles = set(re.findall(r"《([^》]+)》", sentence))
            # Unknown titles are still distinct targets. Ignoring one can attach
            # 'inspired by another film directed by X' to this document's film.
            quoted_movies = set()
            for title in raw_titles:
                quoted_movies.update(self.mentions(title, "Movie"))
            targets = quoted_movies or movie_ids or explicit_movies
            movie = next(iter(targets)) if len(targets) == 1 else None
            # Biography subject ellipsis: '2004年，为《X》中的角色配音'.
            # No use of the synthetic, ambiguous '参演或关联的影片包括' summaries.
            if person_id and movie and len(raw_titles) == 1 and quoted_movies and re.search(
                r"(?:^|\d{4}年[，,]?)\s*(?:他|她|其|并)?(?:曾|再次|首次)?(?:为|在|出演|参演|主演|声演)", sentence
            ) and re.search(r"出演|参演|主演|配音|声演|饰演", sentence):
                before_title = sentence.split("《", 1)[0]
                others = self.mentions(before_title, "Person") - {person_id}
                if not others or ("动画电影" in before_title and "配音" in sentence):
                    add(person_id, "acted_in", movie, match.start(), match.end())
            # Clauses prevent 'A directed, B starred' from assigning A the acting role.
            for clause_match in re.finditer(r"[^，,；;。！？]+", sentence):
                clause = clause_match.group()
                clause_titles = re.findall(r"《([^》]+)》", clause)
                if len(set(clause_titles)) > 1:
                    continue
                preceding_titles = re.findall(r"《([^》]+)》", sentence[:clause_match.end()])
                local_movies = (self.mentions(preceding_titles[-1], "Movie")
                                if preceding_titles else movie_ids)
                if len(local_movies) != 1:
                    continue
                movie = next(iter(local_movies))
                for role, verb in (("directed", r"执导|导演"),
                                   ("acted_in", r"主演|领衔主演|联合主演|出演|参演|配音|声演|饰演|扮演|饰")):
                    persons: set[str] = set()
                    for predicate in re.finditer(verb, clause):
                        before, after = clause[:predicate.start()], clause[predicate.end():]
                        if role == "directed" and predicate.group() == "导演":
                            # Accept '导演：A' or 'A导演的电影', not '最佳导演奖 A'.
                            if after.startswith(("：", ":")):
                                persons |= self.mentions(after, "Person")
                            elif after.startswith("的"):
                                persons |= self.mentions(before, "Person")
                            continue
                        # Only names in the grammatical subject list immediately before the verb.
                        before = re.split(r"由|包括|分别为|是|为|电影|影片", before)[-1]
                        residual = self.pattern.sub("", before)
                        residual = re.sub(r"《[^》]*》|（[^）]*）|\([^)]*\)", "", residual)
                        residual = re.sub(r"演员表?|阵容|领衔|联合|共同|特别|友情|客串|担任|回归|再次|分别|以及|还有|一同|等人|及|与|和|等|将|由", "", residual)
                        if not re.search(r"[\w\u4e00-\u9fff]", residual):
                            persons |= self.mentions(before, "Person")
                        if re.match(r"\s*[：:]", after):
                            persons |= self.mentions(after, "Person")
                    for person in persons:
                        add(person, role, movie, match.start(), match.end())
        return list(result.values())
