from __future__ import annotations

import json
from types import SimpleNamespace

from src.core.types import (
    Answer, Chunk, EdgeMask, Entity, Evidence, Relation, RetrievalConstraints, RetrievalResult,
)
from src.graph.networkx_store import NetworkxGraphStore
from src.methods.hipporag2 import HippoRAG2Method
from src.methods.kg2rag import KG2RAGMethod
from src.methods.library_graphrag import LibraryGraphRAGMethod
from src.methods import available
from src.methods.vector import VectorQAMethod
from src.retrieve.anchors import AnchorResolver
from src.retrieve.dataset_graph import DatasetGraphLoader
from src.retrieve.hipporag2 import HippoRAG2Retriever
from src.retrieve.kg2rag import KG2RAGRetriever
from src.retrieve.registry import RetrievalContext


class FakeSettings:
    def get(self, key, default=None):
        values = {"retrieval.top_k_chunks": 3}
        return values.get(key, default)


class FakeIndex:
    def search(self, question, top_k):
        assert question == "测试问题"
        assert top_k == 2
        return [(Chunk("c1", "d1", "命中文本", 0), 0.9)]


class FakeGenerator:
    def generate(self, question, result):
        assert result.chunks[0].id == "c1"
        return Answer(text="本地向量答案")


class GraphSettings:
    def get(self, key, default=None):
        values = {
            "retrieval.top_k_chunks": 2,
            "kg2rag.seed_chunks": 1,
            "kg2rag.anchor_top_n": 4,
            "kg2rag.max_hops": 2,
            "kg2rag.max_nodes": 20,
            "kg2rag.candidate_chunks": 10,
            "kg2rag.semantic_weight": 0.5,
            "kg2rag.graph_weight": 0.5,
            "kg2rag.hop_decay": 0.8,
            "hipporag2.anchor_top_n": 4,
            "hipporag2.alpha": 0.85,
            "hipporag2.max_iter": 300,
            "hipporag2.tolerance": 1e-10,
            "hipporag2.top_nodes": 10,
            "hipporag2.degree_penalty": 0.5,
            "hipporag2.max_path_hops": 3,
            "hipporag2.graph_weight": 0.85,
            "hipporag2.semantic_weight": 0.15,
        }
        return values.get(key, default)


class FakeGraphIndex:
    def __init__(self, chunks):
        self.chunks = chunks
        self._scores = {"c-film-1": 0.95, "c-film-2": 0.35}

    def search(self, question, top_k):
        return [
            (chunk, self._scores.get(chunk.id, 0.0))
            for chunk in self.chunks[:top_k]
        ]

    def score_chunks(self, question, chunk_ids):
        return {chunk_id: self._scores.get(chunk_id, 0.0) for chunk_id in chunk_ids}


def graph_context():
    chunks = [
        Chunk("c-film-1", "film-1", "导演甲执导影片一，演员乙与演员丙出演。", 0),
        Chunk("c-film-2", "film-2", "导演甲执导影片二，演员乙再次出演。", 0),
    ]
    evidence_1 = Evidence("film-1", "c-film-1", 0, len(chunks[0].text), chunks[0].text)
    evidence_2 = Evidence("film-2", "c-film-2", 0, len(chunks[1].text), chunks[1].text)
    entities = [
        Entity("director", "导演甲", "Person"),
        Entity("actor-b", "演员乙", "Person"),
        Entity("actor-c", "演员丙", "Person"),
        Entity("film-1", "影片一", "Movie", evidences=[evidence_1]),
        Entity("film-2", "影片二", "Movie", evidences=[evidence_2]),
    ]
    relations = [
        Relation("r1", "director", "film-1", "directed", evidences=[evidence_1]),
        Relation("r2", "director", "film-2", "directed", evidences=[evidence_2]),
        Relation("r3", "actor-b", "film-1", "acted_in", evidences=[evidence_1]),
        Relation("r4", "actor-b", "film-2", "acted_in", evidences=[evidence_2]),
        Relation("r5", "actor-c", "film-1", "acted_in", evidences=[evidence_1]),
    ]
    store = NetworkxGraphStore()
    for entity in entities:
        store.upsert_entity(entity)
    for relation in relations:
        store.upsert_relation(relation)
    index = FakeGraphIndex(chunks)
    settings = GraphSettings()
    return RetrievalContext(store, index, settings, AnchorResolver(store))


def test_vector_method_uses_local_index_and_generator():
    method = VectorQAMethod(FakeSettings(), index=FakeIndex(), generator=FakeGenerator())
    answer = method.ask("测试问题", top_k=2)
    assert answer.text == "本地向量答案"
    assert answer.retriever_name == "vector"
    assert answer.debug_info["retrieval"]["backend"] == "local-numpy"


def test_library_context_is_converted_to_project_answer_types():
    item = SimpleNamespace(
        content="片段和三元组",
        metadata={
            "chunk_id": "chunk-1",
            "text": "片段",
            "score": 0.8,
            "nodes": [
                {"id": "n1", "name": "导演甲", "type": "Person"},
                {"id": "n2", "name": "电影乙", "type": "Movie"},
            ],
            "edges": [
                {"id": "r1", "source": "n1", "target": "n2", "type": "directed"}
            ],
        },
    )
    citations, graph = LibraryGraphRAGMethod._convert_context([item])
    assert citations[0].chunk_id == "chunk-1"
    assert {node.name for node in graph.entities} == {"导演甲", "电影乙"}
    assert graph.relations[0].type == "directed"
    assert graph.node_scores["n1"] == 0.8


def test_expected_methods_are_registered_without_restricting_future_plugins():
    assert {"hipporag2", "kg2rag", "library_graphrag", "vector"} <= set(available())


def test_library_method_calls_full_rag_search_and_returns_context():
    item = SimpleNamespace(content="上下文", metadata={"chunk_id": "c1", "text": "上下文"})

    class FakeRag:
        def search(self, **kwargs):
            assert kwargs["query_text"] == "图问题"
            assert kwargs["retriever_config"] == {
                "top_k": 4, "query_params": {"masked_edge_keys": []}
            }
            assert kwargs["return_context"] is True
            return SimpleNamespace(
                answer="库生成答案",
                retriever_result=SimpleNamespace(items=[item]),
            )

    method = LibraryGraphRAGMethod(FakeSettings())
    method._rag = FakeRag()
    answer = method.ask("图问题", top_k=4)
    assert answer.text == "库生成答案"
    assert answer.retriever_name == "library_graphrag"
    assert answer.citations[0].snippet == "上下文"


def test_library_method_passes_query_local_edge_masks():
    class FakeRag:
        def search(self, **kwargs):
            keys = kwargs["retriever_config"]["query_params"]["masked_edge_keys"]
            assert "actor|acted_in|film" in keys
            assert "film|出演|actor" in keys
            return SimpleNamespace(answer="答案", retriever_result=SimpleNamespace(items=[]))

    method = LibraryGraphRAGMethod(FakeSettings())
    method._rag = FakeRag()
    method.ask(
        "图问题", constraints=RetrievalConstraints((EdgeMask("actor", "acted_in", "film"),))
    )


def test_kg2rag_expands_vector_seed_and_reranks_graph_evidence():
    result = KG2RAGRetriever(graph_context()).retrieve("导演甲执导过什么？", top_k=2)
    assert [chunk.id for chunk in result.chunks] == ["c-film-1", "c-film-2"]
    assert {entity.id for entity in result.entities} >= {"director", "film-1", "film-2"}
    assert result.debug_info["expanded_relations"] >= 2
    assert result.debug_info["reranked_chunks"][0]["semantic"] == 1.0


def test_hipporag2_keeps_bridge_path_between_query_entities():
    result = HippoRAG2Retriever(graph_context()).retrieve(
        "演员乙与演员丙通过什么影片关联？", top_k=2
    )
    assert ["actor-b", "film-1", "actor-c"] in result.debug_info["bridge_paths"]
    assert result.debug_info["converged"] is True
    assert result.chunks[0].id == "c-film-1"


def test_local_graph_retrievers_hide_masked_edge_without_mutating_store():
    context = graph_context()
    constraints = RetrievalConstraints((EdgeMask("actor-c", "acted_in", "film-1"),))
    for retriever in (KG2RAGRetriever(context), HippoRAG2Retriever(context)):
        result = retriever.retrieve(
            "演员乙与演员丙通过什么影片关联？", top_k=2, constraints=constraints
        )
        assert not any(relation.id == "r5" for relation in result.relations)
        assert result.debug_info["masked_edge_count"] == 1
    assert any(relation.id == "r5" for relation in context.store.all_relations())


def test_new_methods_are_isolated_qamethod_wrappers():
    class FakeRetriever:
        def retrieve(self, question, top_k=None, year_range=None):
            return RetrievalResult(
                retriever_name="fake",
                chunks=[Chunk("c1", "d1", "证据", 0)],
                debug_info={"trace": "kept"},
            )

    class PassthroughGenerator:
        def generate(self, question, result):
            return Answer(text=question + result.chunks[0].text)

    for cls, expected_name in ((KG2RAGMethod, "kg2rag"), (HippoRAG2Method, "hipporag2")):
        answer = cls(FakeSettings(), retriever=FakeRetriever(), generator=PassthroughGenerator()).ask(
            "问题", top_k=1
        )
        assert answer.text == "问题证据"
        assert answer.retriever_name == expected_name
        assert answer.debug_info["retrieval"] == {"trace": "kept"}


def test_dataset_graph_uses_schema_direction_and_chunk_provenance(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "films.jsonl").write_text(
        json.dumps({"film": {"id": "film", "name": "影片一"}}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (dataset / "actors.jsonl").write_text("", encoding="utf-8")
    (dataset / "directors.jsonl").write_text(
        json.dumps(
            {"person": {"id": "director", "name": "导演甲", "aliases": ["Director A"]}},
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    (dataset / "relations.jsonl").write_text(
        json.dumps(
            {
                "source_id": "film", "source_name": "影片一", "source_type": "影片",
                "relation": "执导", "target_id": "director", "target_name": "导演甲",
            },
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    class LoaderSettings:
        def get(self, key, default=None):
            return default

        def path(self, key):
            assert key == "paths.dataset_dir"
            return dataset

    chunk = Chunk("chunk", "film_film", "《影片一》由导演甲执导。", 10)
    store = DatasetGraphLoader(LoaderSettings(), FakeGraphIndex([chunk])).load()
    relation = store.all_relations()[0]
    assert (relation.head_id, relation.type, relation.tail_id) == (
        "director", "directed", "film"
    )
    assert relation.evidences[0].chunk_id == "chunk"
    assert relation.evidences[0].raw_text == "导演甲"
    assert "Director A" in store.get_entity("director").aliases
