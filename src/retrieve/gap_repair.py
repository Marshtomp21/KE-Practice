"""GapRepair: visible-graph plans, directed text repair and budgeted proof selection."""
from __future__ import annotations

from dataclasses import asdict, replace
from functools import lru_cache
import re
from typing import Optional, Tuple

from ..core.interfaces import Retriever
from ..core.types import EvidenceNode, Relation, RetrievalConstraints, RetrievalResult
from .anchors import anchors_to_debug
from .gap_evidence import EvidenceVerifier
from .corpus_evidence import corpus_record_edges
from .gap_plan import QueryPlan, parse_plan
from .hipporag2 import HippoRAG2Retriever
from .registry import RetrievalContext


def edge_key(edge: Relation) -> tuple[str, str, str]:
    return edge.head_id, edge.type, edge.tail_id


class GapRepairRetriever(Retriever):
    name = "gap_repair"

    def __init__(self, context: RetrievalContext) -> None:
        self.context = context
        self.entities = {e.id: e for e in context.store.all_entities()}
        groups = {}
        for entity in self.entities.values():
            name = re.sub(r"\s*[（(](?:电影|film)[）)]$", "", entity.name, flags=re.I) if entity.type == "Movie" else entity.name
            groups.setdefault((entity.type, name), []).append(entity)
        self.canonical = {e.id: e.id for e in self.entities.values()}
        for group in groups.values():
            identified = [e for e in group if e.id.startswith("Q") and e.id[1:].isdigit()]
            if len(identified) == 1:
                target = identified[0]
                for entity in group:
                    if entity.id.startswith(("NAME:", "WP:")):
                        self.canonical[entity.id] = target.id
                self.entities[target.id] = replace(target, aliases=list(dict.fromkeys(
                    s for e in group for s in e.surface_forms() if s != target.name)))
        self.entities = {key: value for key, value in self.entities.items() if self.canonical[key] == key}
        self.chunks = context.chunk_lookup()
        self.verifier = EvidenceVerifier(self.entities)
        self.fallback = HippoRAG2Retriever(context)
        self.default_top_k = int(context.settings.get("retrieval.top_k_chunks", 6))
        self.search_k = int(context.settings.get("gap_repair.search_top_k", 12))
        self.max_queries = int(context.settings.get("gap_repair.max_queries", 12))
        self.max_rounds = int(context.settings.get("gap_repair.max_rounds", 2))
        self.audit_chunks = int(context.settings.get("gap_repair.audit_chunks", 40))
        # Cache only graph-independent vector hits, never a repaired graph/result.
        self._search = lru_cache(maxsize=256)(
            lambda query: tuple(self.context.index.search(query, top_k=self.search_k)))
        self.mode = context.settings.get("gap_repair.compensation", "adaptive")
        self.repair = bool(context.settings.get("gap_repair.temporary_edges", True))
        self.prune = bool(context.settings.get("gap_repair.prune", True))
        self.enhanced = bool(context.settings.get("gap_repair.frontier_queries", False))
        self.corpus_records = bool(context.settings.get("gap_repair.corpus_records", False))
        self.by_doc = {}
        for chunk in self.chunks.values():
            self.by_doc.setdefault(chunk.doc_id, []).append(chunk)
        self.role_docs = {}
        for doc_id in self.by_doc:
            owner = re.fullmatch(r"person_(actor|director)_(.+)", doc_id)
            if owner:
                key = (owner.group(1), self.canonical.get(owner.group(2), owner.group(2)))
                self.role_docs.setdefault(key, set()).add(doc_id)
        if self.mode not in {"adaptive", "always", "off"}:
            raise ValueError("gap_repair.compensation must be adaptive, always or off")
        if min(self.search_k, self.max_queries, self.max_rounds, self.audit_chunks) < 1:
            raise ValueError("gap_repair search/round/audit budgets must be positive")

    def _scope(self, plan: QueryPlan, edges: list[Relation]) -> list[Relation]:
        roots = set(plan.subjects)
        films = {r.tail_id for r in edges if r.head_id in roots and r.type in {"acted_in", "directed"}}
        if plan.kind == "directors":
            return [r for r in edges if r.tail_id in roots and r.type == "directed"]
        return [r for r in edges if r.type in {"acted_in", "directed"}
                and (r.head_id in roots or r.tail_id in films)]

    def _queries(self, plan: QueryPlan, edges: list[Relation], question: str) -> list[str]:
        names = [self.entities[s].name for s in plan.subjects]
        if plan.kind == "directors":
            return [f"《{names[0]}》 全部导演 执导"]
        if plan.kind == "cofilm":
            queries = [f"{' '.join(names)} 共同出演 电影"]
            roles = ["acted_in", "acted_in"]
        elif plan.kind == "director_actor":
            queries = [f"{names[0]} 执导 {names[1]} 出演 电影"]
            roles = ["directed", "acted_in"]
        else:
            queries = [f"{name} 执导 电影 演员 主演" for name in names]
            roles = ["directed"] * len(names)
        present = {edge_key(r) for r in edges}
        # Query candidate film slots first; these names come only from visible/repaired edges.
        films = sorted({r.tail_id for r in edges if r.head_id in plan.subjects})
        if plan.kind in {"cofilm", "director_actor"}:
            for movie in films:
                for person, role, name in zip(plan.subjects, roles, names):
                    if (person, role, movie) not in present:
                        label = "出演 主演" if role == "acted_in" else "执导 导演"
                        queries.append(f"《{self.entities[movie].name}》 {name} {label}")
        else:
            for movie in films:
                queries.append(f"《{self.entities[movie].name}》 演员 主演 配音")
        # Open slots can recover a film with *both* incident edges missing.
        queries.extend(f"{name} {'出演 主演' if role == 'acted_in' else '执导 导演'} 电影"
                       for name, role in zip(names, roles))
        if self.enhanced:
            root_queries = [f"{name} {'参演' if role == 'acted_in' else '执导'} 影片 片单"
                            for name, role in zip(names, roles)]
            # Candidate actors on one director's side / below the count threshold
            # are the useful frontier, not arbitrary alphabetic film titles.
            if plan.kind in {"repeated_cast", "director_overlap"}:
                proof_ids = set(plan.paths(edges))
                counts = {}
                for edge in edges:
                    if edge.type == "acted_in" and edge.head_id not in proof_ids:
                        counts.setdefault(edge.head_id, set()).add(edge.tail_id)
                frontier = sorted(counts, key=lambda person: (-len(counts[person]), person))
                root_queries.extend(f"{self.entities[p].name} 参演 影片 片单" for p in frontier)
            queries = root_queries[:max(2, self.max_queries // 2)] + queries
        return list(dict.fromkeys(queries))

    def retrieve(self, question: str, top_k: Optional[int] = None,
                 year_range: Optional[Tuple[Optional[int], Optional[int]]] = None,
                 constraints: Optional[RetrievalConstraints] = None) -> RetrievalResult:
        limit = self.default_top_k if top_k is None else top_k
        if limit < 1:
            raise ValueError("top_k must be positive")
        anchors = self.context.anchors.resolve(question, top_n=12)
        plan = parse_plan(question, list({self.canonical[a.entity.id]: self.entities[self.canonical[a.entity.id]]
                                         for a in anchors}.values()))
        if plan is None or year_range is not None:
            result = self.fallback.retrieve(question, top_k=limit, year_range=year_range, constraints=constraints)
            result.retriever_name = self.name
            result.debug_info.update({"gap_detected": False, "compensation_triggered": False,
                                      "gap_repair_status": "unsupported_query_fallback"})
            return result

        # The mask is used only by this visibility boundary. Neither its endpoints,
        # its length nor oracle queries are exposed to the planner/verifier.
        view = constraints or RetrievalConstraints()
        visible = [replace(r, head_id=self.canonical[r.head_id], tail_id=self.canonical[r.tail_id])
                   for r in self.context.store.all_relations()
                   if not view.masks(self.canonical[r.head_id], r.type, self.canonical[r.tail_id])
                   and not view.masks(r.head_id, r.type, r.tail_id)]
        scoped = self._scope(plan, visible)
        visible_keys = {edge_key(r) for r in visible}
        active = {edge_key(r): r for r in scoped}
        pool = {}
        relevance: dict[str, float] = {}
        checked: set[str] = set()
        repaired: dict[tuple[str, str, str], Relation] = {}
        queried: list[str] = []
        retrieved_by_compensation: set[str] = set()
        initial_paths = plan.paths(scoped)
        corpus_scope = bool(re.search(r"当前语料|当前数据集|本数据集|语料收录", question))
        # Enabling this policy explicitly scopes role-record answers to this corpus.
        allow_records = self.corpus_records
        record_owners: set[str] = set()
        complete_records: dict[str, tuple[set[str], set[str]]] = {}

        def ingest(chunks, compensation=False):
            additions = 0
            for chunk, score in chunks:
                pool[chunk.id] = chunk
                relevance[chunk.id] = max(relevance.get(chunk.id, 0.0), float(score))
                if compensation:
                    retrieved_by_compensation.add(chunk.id)
                if chunk.id in checked:
                    continue
                checked.add(chunk.id)
                verified = self.verifier.verify(chunk)
                if allow_records:
                    records = corpus_record_edges(chunk, self.verifier, self.canonical)
                    verified.extend(records)
                    record_owners.update(r.head_id for r in records)
                    if records:
                        raw = records[0].evidences[0].raw_text
                        titles = re.findall(r"《([^》]+)》", raw)
                        if all(len(self.verifier.mentions(t, "Movie")) == 1 for t in titles):
                            films, sources = complete_records.setdefault(records[0].head_id, (set(), set()))
                            films.update(r.tail_id for r in records)
                            sources.add(chunk.id)
                relevant_keys = {edge_key(r) for r in self._scope(plan, [*active.values(), *verified])}
                for relation in verified:
                    # Only query-relevant role edges enter the temporary view.
                    if edge_key(relation) not in relevant_keys or not self.repair:
                        continue
                    key = edge_key(relation)
                    if key in visible_keys:
                        continue
                    if key not in repaired:
                        repaired[key] = relation
                        active[key] = relation
                        additions += 1
                    else:
                        previous = repaired[key]
                        if not any(e.chunk_id == chunk.id for e in previous.evidences):
                            previous.evidences.extend(relation.evidences)
                # A repaired first hop unlocks existing second-hop edges. They
                # still come exclusively from the masked, read-only graph view.
                active.update({edge_key(r): r for r in self._scope(plan, [*visible, *repaired.values()])})
            return additions

        # Audit only graph-linked text, not the entire corpus or structured film records.
        audit_ids = list(dict.fromkeys(e.chunk_id for r in scoped for e in r.evidences))[:self.audit_chunks]
        ingest([(self.chunks[c], 0.5) for c in audit_ids if c in self.chunks])
        provisional = []
        if not initial_paths:
            provisional.append({"reason": "no_complete_proof", "target_type": plan.target_type,
                                "subjects": list(plan.subjects)})
        if repaired:
            provisional.append({"reason": "text_graph_disagreement", "count": len(repaired)})
        # A found answer does not certify a complete set. Enumeration requires an audit.
        if not plan.boolean:
            provisional.append({"reason": "set_completeness_unverified", "target_type": plan.target_type})
        trigger = self.mode == "always" or (self.mode == "adaptive" and bool(provisional))
        if self.mode == "off":
            trigger = False
            repaired.clear()
            active = {edge_key(r): r for r in scoped}
        if trigger:
            for _ in range(self.max_rounds):
                queries = [q for q in self._queries(plan, list(active.values()), question) if q not in queried]
                # Reserve part of the total budget for another expansion round.
                budget = min(len(queries), max(1, self.max_queries // self.max_rounds), self.max_queries - len(queried))
                if budget <= 0:
                    break
                additions = 0
                for query in queries[:budget]:
                    queried.append(query)
                    hits = self._search(query)
                    additions += ingest(hits, True)
                    if self.enhanced:
                        # Document-local expansion is bounded to retrieved documents;
                        # no global relation extraction or gold-document lookup.
                        siblings = {}
                        for chunk, score in hits:
                            owner = re.fullmatch(r"person_(actor|director)_(.+)", chunk.doc_id)
                            documents = {chunk.doc_id}
                            if owner:
                                key = (owner.group(1), self.canonical.get(owner.group(2), owner.group(2)))
                                documents = self.role_docs.get(key, documents)
                            for doc_id in documents:
                                for sibling in self.by_doc.get(doc_id, []):
                                    if "是数据集中的" in sibling.text:
                                        siblings[sibling.id] = (sibling, score)
                        additions += ingest(list(siblings.values()), True)
                if not additions and len(queries) <= budget:
                    break

        proofs = plan.paths(list(active.values()))
        # Greedy proof-bundle set cover. Every accepted temporary edge must have a
        # selected source chunk; no hidden supporting chunks outside top_k.
        selected: set[str] = set()
        chosen: dict[str, list[Relation]] = {}

        def support(path):
            required: set[str] = set()
            for edge in path:
                ids = {e.chunk_id for e in edge.evidences if e.chunk_id in self.chunks}
                if not ids:
                    if edge.attributes.get("temporary"):
                        return None
                    continue
                best = min(ids, key=lambda c: (c not in selected, c not in required,
                                               -relevance.get(c, 0.0), c))
                required.add(best)
            return required

        while len(chosen) < len(proofs):
            options = []
            for answer_id, alternatives in proofs.items():
                if answer_id in chosen:
                    continue
                for path in alternatives:
                    required = support(path)
                    if required is None or len(selected | required) > limit:
                        continue
                    new = required - selected
                    options.append((len(new), -sum(relevance.get(c, 0.0) for c in required),
                                    answer_id, tuple(r.id for r in path), path, required))
            if not options:
                break
            _, _, answer_id, _, path, required = min(options, key=lambda x: x[:4])
            chosen[answer_id] = path
            selected.update(required)

        kept = {edge_key(r): r for path in chosen.values() for r in path}
        if not chosen:
            # Return evidence for uncertainty, without inventing a negative answer.
            selected = set(sorted(pool, key=lambda c: (-relevance.get(c, 0.0), c))[:limit])
        negative_assessment = None
        if allow_records and corpus_scope and plan.kind == "cofilm" and plan.boolean and not proofs:
            if all(s in complete_records for s in plan.subjects):
                left, right = [complete_records[s][0] for s in plan.subjects]
                sources = set().union(*(complete_records[s][1] for s in plan.subjects))
                if queried and not left & right and len(sources) <= limit:
                    selected = sources
                    # This is an observation after bilateral text audit, NOT a
                    # certificate that the incomplete corpus contains every fact.
                    negative_assessment = {"scope": "retrieved_corpus_evidence",
                                           "exhaustive": False,
                                           "source_chunk_ids": sorted(sources),
                                           "observed_film_sets": {s: sorted(complete_records[s][0]) for s in plan.subjects}}
        if not self.prune:
            kept = dict(active)
            selected.update(sorted(pool, key=lambda c: (-relevance.get(c, 0.0), c))[:max(0, limit - len(selected))])
        temporary = [r for r in kept.values() if r.attributes.get("temporary")
                     and any(e.chunk_id in selected for e in r.evidences)]
        graph_edges = [r for r in kept.values() if not r.attributes.get("temporary")]
        entity_ids = set(plan.subjects) | {s for r in kept.values() for s in r.endpoints}
        ordered = sorted(selected, key=lambda c: (-relevance.get(c, 0.0), c))
        nodes = []
        for cid in ordered:
            supporting = [r for r in temporary if any(e.chunk_id == cid for e in r.evidences)]
            if supporting:
                nodes.append(EvidenceNode("evidence:" + cid, self.chunks[cid],
                             sorted({s for r in supporting for s in r.endpoints}),
                             [{**r.to_dict(), "evidences": [e.to_dict() for e in r.evidences
                                                             if e.chunk_id == cid]} for r in supporting]))
        debug = {
            "method": self.name, "plan": asdict(plan), "anchors": anchors_to_debug(anchors),
            "provisional_gaps": provisional,
            "gap_suspected": bool(provisional), "gap_detected": bool(repaired),
            "gap_detection_stage": "verified_text_graph_disagreement",
            "compensation_triggered": bool(queried), "compensation_queries": queried,
            "compensation_documents": sorted({self.chunks[c].doc_id for c in selected
                                               if c in retrieved_by_compensation}),
            "temporary_relations": [
                {"head_id": r.head_id, "tail_id": r.tail_id, "relation": r.type,
                 "supporting_documents": sorted({e.doc_id for e in r.evidences if e.chunk_id in selected}),
                 "attributes": r.attributes,
                 "evidences": [e.to_dict() for e in r.evidences if e.chunk_id in selected]}
                for r in temporary],
            "verified_missing_edge_count": len(repaired), "verified_chunk_count": len(checked),
            "answer_ids": sorted(chosen), "candidate_answer_count": len(proofs),
            "answer_paths": {key: [edge_key(r) for r in path] for key, path in chosen.items()},
            "budget_exhausted": len(chosen) < len(proofs),
            "set_completeness": "not_certified_open_world",
            "corpus_record_policy": allow_records,
            "corpus_record_owners": sorted(record_owners),
            "corpus_record_completeness": {s: {"films": sorted(complete_records[s][0]),
                                               "sources": sorted(complete_records[s][1])}
                                           for s in plan.subjects if s in complete_records},
            "negative_assessment": negative_assessment,
            "evidence_tiers": {"explicit_role_span": sum(r.attributes.get("verifier") != "corpus_role_record" for r in temporary),
                               "dataset_assertion": sum(r.attributes.get("verifier") == "corpus_role_record" for r in temporary)},
            "status": "supported_answers" if chosen else "unresolved",
            "selection": "greedy_proof_bundle_cover" if self.prune else "unpruned",
        }
        return RetrievalResult(self.name, [self.chunks[c] for c in ordered],
                               [self.entities[e] for e in sorted(entity_ids) if e in self.entities],
                               graph_edges, {c: relevance.get(c, 0.0) for c in ordered}, debug,
                               evidence_nodes=nodes)
