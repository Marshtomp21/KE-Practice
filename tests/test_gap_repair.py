from pathlib import Path

import pytest

from src.core.config import Settings, load_settings
from src.core.types import Chunk, EdgeMask, Entity, Evidence, Relation, RetrievalConstraints
from src.graph.networkx_store import NetworkxGraphStore
from src.methods.gap_repair import GapRepairMethod
from src.retrieve.gap_evidence import EvidenceVerifier
from src.retrieve.gap_plan import QueryPlan
from src.retrieve.gap_repair import GapRepairRetriever
from src.retrieve.registry import RetrievalContext


class Index:
    def __init__(self, chunks):
        self.chunks = chunks
        self.queries = []

    def search(self, query, top_k):
        self.queries.append(query)
        return [(c, 0.9) for c in self.chunks[:top_k]]


def make_context(text="《影片甲》由导演甲执导，王俊凯与苗苗共同主演。", edges=True, extra=None):
    config = {**load_settings().as_dict(), "gap_repair": {"max_queries": 4, "max_rounds": 2}}
    if extra:
        config["gap_repair"].update(extra)
    settings = Settings(config, Path("test.yaml"))
    chunk = Chunk("c1", "film_f1", text, 17, {"title": "影片甲"})
    evidence = Evidence(chunk.doc_id, chunk.id, 17, chunk.char_end, chunk.text)
    store = NetworkxGraphStore()
    for entity in [Entity("a", "王俊凯", "Person"), Entity("b", "苗苗", "Person"),
                   Entity("d", "导演甲", "Person"), Entity("f1", "影片甲", "Movie")]:
        store.upsert_entity(entity)
    if edges:
        for person, kind in [("a", "acted_in"), ("b", "acted_in"), ("d", "directed")]:
            store.upsert_relation(Relation(person, person, "f1", kind, evidences=[evidence]))
    return RetrievalContext.assemble(store, Index([chunk]), settings)


QUESTION = "王俊凯与苗苗共同出演了哪些电影？"


def test_repairs_both_missing_edges_and_never_persists_or_uses_oracle():
    context = make_context()
    before = [r.to_dict() for r in context.store.all_relations()]
    masks = (EdgeMask("a", "acted_in", "f1"), EdgeMask("b", "acted_in", "f1"))
    retriever = GapRepairRetriever(context)
    result = retriever.retrieve(QUESTION, constraints=RetrievalConstraints(masks, ("ORACLE_SECRET",)))
    assert result.debug_info["answer_ids"] == ["f1"]
    assert result.debug_info["gap_detected"]
    assert len(result.debug_info["temporary_relations"]) == 2
    assert not result.relations
    assert result.evidence_nodes[0].chunk.id == "c1"
    assert all("ORACLE_SECRET" not in q for q in context.index.queries)
    assert [r.to_dict() for r in context.store.all_relations()] == before
    assert not retriever.retrieve(QUESTION).debug_info["gap_detected"]


def test_physical_deletion_matches_masked_view():
    masked = make_context()
    deleted = make_context(edges=False)
    # An unrelated director edge must not change the actor query.
    r1 = GapRepairRetriever(masked).retrieve(QUESTION, constraints=RetrievalConstraints(
        (EdgeMask("a", "acted_in", "f1"), EdgeMask("b", "acted_in", "f1"))))
    r2 = GapRepairRetriever(deleted).retrieve(QUESTION)
    assert r1.debug_info["answer_ids"] == r2.debug_info["answer_ids"]
    assert r1.debug_info["compensation_queries"] == r2.debug_info["compensation_queries"]


@pytest.mark.parametrize("text", [
    "王俊凯与苗苗一起出席电影讨论会。", "王俊凯与苗苗没有共同出演《影片甲》。",
    "《影片甲》原定由王俊凯与苗苗主演。", "王俊凯采访了主演苗苗。",
    "《影片甲》由王俊凯执导，苗苗主演。",
    "《影片甲》的灵感来自王俊凯与苗苗主演的《另一部不在图中的电影》。",
    "王俊凯与苗苗观看了主演参加的电影活动。",
])
def test_cooccurrence_negation_and_wrong_roles_do_not_create_a_complete_path(text):
    context = make_context(text, edges=False)
    result = GapRepairRetriever(context).retrieve(QUESTION)
    assert result.debug_info["answer_ids"] == []


def test_exact_spans_and_aliases():
    context = make_context("《影片甲》由小凯与苗苗主演。", edges=False)
    entities = {e.id: e for e in context.store.all_entities()}
    entities["a"].aliases.append("小凯")
    edges = EvidenceVerifier(entities).verify(context.index.chunks[0])
    assert {e.head_id for e in edges} == {"a", "b"}
    chunk = context.index.chunks[0]
    for edge in edges:
        ev = edge.evidences[0]
        assert chunk.text[ev.char_start - chunk.char_offset:ev.char_end - chunk.char_offset] == ev.raw_text


def test_intro_facts_before_a_sequel_mention_remain_verifiable():
    context = make_context("《影片甲》是电影，由导演甲执导，王俊凯与苗苗主演，其续作是《未知续集》。", edges=False)
    result = GapRepairRetriever(context).retrieve(QUESTION)
    assert result.debug_info["answer_ids"] == ["f1"]


def test_biography_subject_ellipsis_has_local_evidence():
    context = make_context(edges=False)
    verifier = EvidenceVerifier({e.id: e for e in context.store.all_entities()})
    chunk = Chunk("bio", "person_actor_a", "2004年，为动画电影《影片甲》中的角色配音。", 100)
    assert [(r.head_id, r.type, r.tail_id) for r in verifier.verify(chunk)] == [("a", "acted_in", "f1")]


def test_ambiguous_synthetic_association_is_not_an_acting_fact():
    context = make_context("王俊凯是数据集中的演员。参演或关联的影片包括：《影片甲》。", edges=False)
    verifier = EvidenceVerifier({e.id: e for e in context.store.all_entities()})
    assert not verifier.verify(context.index.chunks[0])


def test_temp_nodes_reach_the_api_graph():
    from src.api.server import _graph_payload
    context = make_context(edges=False)
    answer = GapRepairMethod(context.settings, retriever=GapRepairRetriever(context), generator=object()).ask(QUESTION)
    payload = _graph_payload(answer)
    assert any(n["type"] == "EvidenceNode" for n in payload["nodes"])
    assert any(e["type"] == "supports" for e in payload["edges"])


def test_count_requires_distinct_films_and_director_actor_join():
    edges = [Relation("d1", "d", "f1", "directed"), Relation("a1", "a", "f1", "acted_in")]
    plan = QueryPlan("repeated_cast", ("d",), "Person", 2)
    assert not plan.paths(edges + edges)
    edges += [Relation("d2", "d", "f2", "directed"), Relation("a2", "a", "f2", "acted_in")]
    assert set(plan.paths(edges)) == {"a"}
    assert set(QueryPlan("director_actor", ("d", "a"), "Movie").paths(edges)) == {"f1", "f2"}


def test_budget_does_not_claim_paths_supported_by_unselected_chunks():
    context = make_context(edges=False)
    context.index.chunks = [Chunk("c1", "film_f1", "《影片甲》由王俊凯主演。", 0),
                            Chunk("c2", "film_f1", "《影片甲》由苗苗主演。", 100)]
    retriever = GapRepairRetriever(context)
    result = retriever.retrieve(QUESTION, top_k=1)
    assert len(result.chunks) <= 1
    assert not result.debug_info["answer_ids"]
    assert result.debug_info["budget_exhausted"]
    assert not result.debug_info["temporary_relations"]
    assert retriever.retrieve(QUESTION, top_k=2).debug_info["answer_ids"] == ["f1"]


def test_answer_and_evidence_nodes_are_serializable_and_cited():
    context = make_context(edges=False)
    method = GapRepairMethod(context.settings, retriever=GapRepairRetriever(context), generator=object())
    answer = method.ask(QUESTION)
    assert "《影片甲》[S1]" in answer.text
    assert not answer.subgraph.relations  # Recovered edges are not visible graph edges.
    assert answer.to_dict()["subgraph"]["evidence_nodes"]
    context.settings.as_dict()["retrieval"] = {"max_context_chars": 1}
    answer = method.ask(QUESTION)
    assert "影片甲" not in answer.text
    assert not answer.debug_info["retrieval"]["temporary_relations"]


def test_ablation_without_compensation_does_not_repair():
    context = make_context(edges=False, extra={"compensation": "off"})
    result = GapRepairRetriever(context).retrieve(QUESTION)
    assert not result.debug_info["answer_ids"]
    assert not context.index.queries


def test_unknown_negative_is_not_asserted_as_no_answer():
    context = make_context("两人在访谈中讨论电影。", edges=False)
    answer = GapRepairMethod(context.settings, retriever=GapRepairRetriever(context), generator=object()).ask(QUESTION)
    assert "无法确定" in answer.text
    assert not answer.debug_info["retrieval"]["gap_detected"]


def test_role_record_policy_is_explicit_and_provenance_is_weaker():
    from src.retrieve.corpus_evidence import corpus_record_edges
    context = make_context(edges=False)
    chunk = Chunk("record", "person_actor_a", "王俊凯是数据集中的演员。参演或关联的影片包括：《影片甲》。", 0)
    verifier = EvidenceVerifier({e.id: e for e in context.store.all_entities()})
    assert not verifier.verify(chunk)
    edges = corpus_record_edges(chunk, verifier)
    assert len(edges) == 1
    assert edges[0].attributes["evidence_tier"] == "dataset_assertion"
    assert edges[0].evidences[0].confidence == 0.65
    assert not corpus_record_edges(Chunk("x", "untrusted", chunk.text, 0), verifier)


def test_api_does_not_upgrade_weak_record_confidence():
    from src.api.server import _graph_payload
    context = make_context(edges=False, extra={"corpus_records": True})
    context.index.chunks = [
        Chunk("ra", "person_actor_a", "王俊凯是数据集中的演员。参演或关联的影片包括：《影片甲》。", 0),
        Chunk("rb", "person_actor_b", "苗苗是数据集中的演员。参演或关联的影片包括：《影片甲》。", 0),
    ]
    answer = GapRepairMethod(context.settings, retriever=GapRepairRetriever(context), generator=object()).ask(QUESTION)
    nodes = [n for n in _graph_payload(answer)["nodes"] if n["type"] == "EvidenceNode"]
    assert nodes and all(n["type_label"] == "片单推断证据" for n in nodes)
    assert all(n["evidences"][0]["confidence"] == 0.65 for n in nodes)


def test_corpus_negative_is_a_bounded_observation_with_two_disjoint_records():
    context = make_context(edges=False, extra={"corpus_records": True})
    context.store.upsert_entity(Entity("f2", "影片乙", "Movie"))
    context.index.chunks = [
        Chunk("ra", "person_actor_a", "王俊凯是数据集中的演员。参演或关联的影片包括：《影片甲》。", 0),
        Chunk("rb", "person_actor_b", "苗苗是数据集中的演员。参演或关联的影片包括：《影片乙》。", 0),
    ]
    method = GapRepairMethod(context.settings, retriever=GapRepairRetriever(context), generator=object())
    answer = method.ask("在当前语料中，王俊凯与苗苗是否共同出演过电影？")
    assert "未发现共同出演" in answer.text
    assert len(answer.citations) == 2
    assert answer.debug_info["retrieval"]["negative_assessment"]["exhaustive"] is False
    assert "无法确定" in method.ask("在当前语料中，王俊凯与苗苗是否共同出演过电影？", top_k=1).text


def test_incomplete_record_cannot_certify_negative():
    context = make_context(edges=False, extra={"corpus_records": True})
    context.index.chunks = [Chunk("ra", "person_actor_a", "王俊凯是数据集中的演员。参演或关联的影片包括：《影片甲》。", 0),
                            Chunk("rb", "person_actor_b", "苗苗是数据集中的演员。参演或关联的影片包括：《未知影片》。", 0)]
    result = GapRepairRetriever(context).retrieve("在当前语料中，王俊凯与苗苗是否共同出演过电影？")
    assert result.debug_info["negative_assessment"] is None


def test_query_cache_never_caches_repaired_graph():
    context = make_context(extra={"frontier_queries": True})
    retriever = GapRepairRetriever(context)
    masks = RetrievalConstraints((EdgeMask("b", "acted_in", "f1"),))
    masked = retriever.retrieve(QUESTION, constraints=masks)
    complete = retriever.retrieve(QUESTION)
    assert masked.debug_info["temporary_relations"]
    assert not complete.debug_info["temporary_relations"]
    assert {r.head_id for r in complete.relations} == {"a", "b"}
