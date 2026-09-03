from __future__ import annotations

from types import SimpleNamespace

from src.core.types import Answer, Chunk
from src.methods.library_graphrag import LibraryGraphRAGMethod
from src.methods import available
from src.methods.vector import VectorQAMethod


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


def test_only_current_methods_are_registered():
    assert available() == ["library_graphrag", "vector"]


def test_library_method_calls_full_rag_search_and_returns_context():
    item = SimpleNamespace(content="上下文", metadata={"chunk_id": "c1", "text": "上下文"})

    class FakeRag:
        def search(self, **kwargs):
            assert kwargs["query_text"] == "图问题"
            assert kwargs["retriever_config"] == {"top_k": 4}
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
