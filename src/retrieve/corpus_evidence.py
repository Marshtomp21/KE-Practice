"""Optional interpretation of role-scoped filmography records in this corpus.

The importer materializes actor/director filmography as text. These are weaker
than explicit natural-language role assertions: their wording says 'or related'.
Only an explicit corpus-record policy can turn them into typed membership edges.
The provenance/assumption stays attached to every edge and must be reported.
"""
from __future__ import annotations

import hashlib
import re

from ..core.types import Chunk, Evidence, Relation
from .gap_evidence import EvidenceVerifier


def corpus_record_edges(chunk: Chunk, verifier: EvidenceVerifier, canonical: dict[str, str] | None = None) -> list[Relation]:
    owner = re.fullmatch(r"person_(actor|director)_(.+)", chunk.doc_id)
    if not owner:
        return []
    person = (canonical or {}).get(owner.group(2), owner.group(2))
    if person not in verifier.entities:
        return []
    role = "acted_in" if owner.group(1) == "actor" else "directed"
    label, verb = ("演员", "参演") if role == "acted_in" else ("导演", "执导")
    entity = verifier.entities[person]
    names = "|".join(re.escape(s) for s in entity.surface_forms())
    pattern = re.compile(r"(?:" + names + r")是数据集中的" + label + r"。" + verb + r"或关联的影片包括：(?P<films>(?:《[^》]+》[、，]?)+)。")
    result = []
    for match in pattern.finditer(chunk.text):
        for title in re.finditer(r"《([^》]+)》", match.group("films")):
            movies = verifier.mentions(title.group(1), "Movie")
            if len(movies) != 1:
                continue
            movie = next(iter(movies))
            token = f"{person}|{role}|{movie}|{chunk.id}"
            evidence = Evidence(chunk.doc_id, chunk.id, chunk.char_offset + match.start(),
                                chunk.char_offset + match.end(), match.group(), 0.65)
            result.append(Relation("corpus-record-" + hashlib.sha1(token.encode()).hexdigest(),
                                   person, movie, role,
                                   attributes={"temporary": True, "verifier": "corpus_role_record",
                                               "evidence_tier": "dataset_assertion",
                                               "assumption": "role_scoped_filmography_membership_not_natural_language_entailment"},
                                   evidences=[evidence]))
    return result
