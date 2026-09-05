"""Small, explicit movie-domain query plans; no benchmark metadata is accepted."""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..core.types import Entity, Relation


@dataclass(frozen=True)
class QueryPlan:
    kind: str
    subjects: tuple[str, ...]
    target_type: str
    minimum: int = 1
    boolean: bool = False

    def paths(self, relations: list[Relation]) -> dict[str, list[list[Relation]]]:
        """Return alternative proofs, with distinct films for counting queries."""
        acted: dict[str, dict[str, Relation]] = {}
        directed: dict[str, dict[str, Relation]] = {}
        for edge in relations:
            table = acted if edge.type == "acted_in" else directed if edge.type == "directed" else None
            if table is not None:
                table.setdefault(edge.head_id, {})[edge.tail_id] = edge
        if self.kind == "directors":
            movie = self.subjects[0]
            return {p: [[films[movie]]] for p, films in directed.items() if movie in films}
        if self.kind in {"cofilm", "director_actor"}:
            left = (directed if self.kind == "director_actor" else acted).get(self.subjects[0], {})
            right = acted.get(self.subjects[1], {})
            return {m: [[left[m], right[m]]] for m in sorted(left.keys() & right.keys())}
        result: dict[str, list[list[Relation]]] = {}
        left = directed.get(self.subjects[0], {})
        if self.kind == "repeated_cast":
            for person, movies in acted.items():
                common = sorted(left.keys() & movies.keys())
                if len(common) >= self.minimum:
                    # A proof contains at least `minimum` distinct films, never duplicate edges.
                    from itertools import combinations, islice
                    result[person] = [
                        [edge for m in group for edge in (left[m], movies[m])]
                        for group in islice(combinations(common, self.minimum), 64)
                    ]
        elif self.kind == "director_overlap":
            right = directed.get(self.subjects[1], {})
            for person, movies in acted.items():
                a, b = sorted(left.keys() & movies.keys()), sorted(right.keys() & movies.keys())
                if a and b:
                    result[person] = [
                        list({e.id: e for e in (left[x], movies[x], right[y], movies[y])}.values())
                        for x in a for y in b
                    ][:64]
        return result


def parse_plan(question: str, entities: list[Entity]) -> QueryPlan | None:
    """Conservative grammar: unsupported constraints fall back to ordinary retrieval."""
    if re.search(r"(?:19|20)\d{2}|之前|之后|以前|以后|票房|获奖", question):
        return None
    mentioned = []
    for entity in entities:
        positions = [question.find(s) for s in entity.surface_forms() if len(s) >= 2 and s in question]
        if positions:
            mentioned.append((min(positions), entity))
    mentioned.sort(key=lambda pair: pair[0])
    people = [e for _, e in mentioned if e.type == "Person"]
    movies = [e for _, e in mentioned if e.type == "Movie"]
    if len(movies) == 1 and not people and "导演" in question:
        return QueryPlan("directors", (movies[0].id,), "Person")
    if len(people) == 1 and re.search(r"至少|重复|两次|多次", question) and "影片" in question:
        match = re.search(r"至少(?:出现过)?([二两三四五六七八九十\d]+)", question)
        raw = match.group(1) if match else "两"
        minimum = int(raw) if raw.isdigit() else {"二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
                                                       "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}.get(raw, 2)
        return QueryPlan("repeated_cast", (people[0].id,), "Person", max(2, minimum))
    if len(people) != 2:
        return None
    if "哪些演员" in question and ("执导" in question or "导演" in question):
        return QueryPlan("director_overlap", tuple(p.id for p in people), "Person")
    if "执导" in question:
        # Bind the director by local syntax, never by inspecting their stored edges.
        directors = [p for p in people if any(re.search(re.escape(s) + r"(?:所)?执导", question)
                                             for s in p.surface_forms())]
        if len(directors) == 1:
            director = directors[0]
            actor = next(p for p in people if p.id != director.id)
            return QueryPlan("director_actor", (director.id, actor.id), "Movie")
        return None
    if re.search(r"共同|一起|合作|关联", question):
        return QueryPlan("cofilm", tuple(p.id for p in people), "Movie", boolean="是否" in question)
    return None
