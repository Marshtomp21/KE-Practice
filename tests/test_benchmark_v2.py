from pathlib import Path

from eval.benchmark_methods import BenchmarkHybridMethod
from src.core.types import (
    Answer, Chunk, Citation, EdgeMask, Entity, Relation, RetrievalConstraints,
    RetrievalResult, Subgraph,
)
from src.evaluation import BenchmarkScorer, load_benchmark
from src.evaluation.benchmark import answer_body


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "eval" / "benchmark_v2" / "questions.yaml"


def test_frozen_benchmark_has_40_auditable_questions():
    payload = load_benchmark(BENCHMARK)
    assert len(payload["questions"]) == 40
    assert payload["benchmark"]["status"] == "frozen"
    assert sum(item["split"] == "dev" for item in payload["questions"]) == 8
    assert sum(item["split"] == "test" for item in payload["questions"]) == 32
    for question in payload["questions"]:
        assert question["gold_documents"]
        assert question["gold_evidence"]
        assert all(item["char_end"] > item["char_start"] >= 0 for item in question["gold_evidence"])


def test_gap_benchmark_has_recoverable_missing_edges_and_controls():
    payload = load_benchmark(BENCHMARK)
    conditions = [item["graph_perturbation"]["condition"] for item in payload["questions"]]
    assert conditions.count("critical_edge_missing") == 22
    assert conditions.count("count_support_missing") == 8
    assert conditions.count("complete_control") == 4
    assert conditions.count("negative_control") == 6
    for question in payload["questions"]:
        gap = question["graph_perturbation"]
        if gap["expected_gap"]:
            assert gap["masked_edges"]
            assert gap["compensation_gold_documents"]
            assert gap["compensation_gold_evidence"]
            assert gap["oracle_queries"]


def test_raw_evidence_lines_cannot_leak_gold_into_answer_score():
    payload = load_benchmark(BENCHMARK)
    question = next(item for item in payload["questions"] if item["id"] == "v2-cofilm-01")
    gold_names = "、".join(item["name"] for item in question["gold_answers"])
    answer = Answer(text=f"材料不足，无法作答。\n[S1] 原文：{gold_names}")
    metrics = BenchmarkScorer(payload).score(question, answer)
    assert gold_names not in answer_body(answer.text)
    assert metrics["answer"]["f1"] == 0.0


def test_set_f1_penalizes_extra_entity_and_accepts_aliases():
    payload = load_benchmark(BENCHMARK)
    question = next(item for item in payload["questions"] if item["id"] == "v2-cofilm-04")
    # 两个 gold + 一个数据集内错误片名；2046 使用去后缀别名。
    answer = Answer(text="《2046》、《花样年华》和《无间道》")
    metrics = BenchmarkScorer(payload).score(question, answer)["answer"]
    assert metrics["recall"] == 1.0
    assert metrics["precision"] < 1.0
    assert metrics["exact_match"] == 0.0


def test_no_answer_requires_specific_denial_not_insufficient_context():
    payload = load_benchmark(BENCHMARK)
    question = next(item for item in payload["questions"] if item["kind"] == "hard_negative")
    scorer = BenchmarkScorer(payload)
    refused = scorer.score(question, Answer(text="材料不足，无法判断。"))["answer"]
    denied = scorer.score(question, Answer(text="结论：无共同出演的影片。"))["answer"]
    assert refused["f1"] == 0.0
    assert denied["f1"] == 1.0


def test_relation_paths_accept_library_chinese_labels_and_stable_ids():
    payload = load_benchmark(BENCHMARK)
    question = next(item for item in payload["questions"] if item["id"] == "v2-cofilm-05")
    path = question["gold_paths"][0]
    entities = []
    relations = []
    for index, edge in enumerate(path["edges"]):
        head = payload["entity_catalog"][edge["head_id"]]
        tail = payload["entity_catalog"][edge["tail_id"]]
        entities.extend([
            Entity(id=head["id"], name=head["name"], type=head["type"]),
            Entity(id=tail["id"], name=tail["name"], type=tail["type"]),
        ])
        relations.append(Relation(
            id=f"r{index}", head_id=edge["tail_id"], tail_id=edge["head_id"], type="出演"
        ))
    answer = Answer(text="《无间道》", subgraph=Subgraph(entities=entities, relations=relations))
    retrieval = BenchmarkScorer(payload).score(question, answer)["retrieval"]
    assert retrieval["relation_recall"] == 1.0
    assert retrieval["path_complete_rate"] == 1.0


def test_temporary_relation_only_repairs_path_when_gold_document_supports_it():
    payload = load_benchmark(BENCHMARK)
    question = next(item for item in payload["questions"] if item["id"] == "v2-cofilm-01")
    gap = question["graph_perturbation"]
    masked = gap["masked_edges"][0]
    visible_edges = [
        edge for path in question["gold_paths"] for edge in path["edges"]
        if edge != masked
    ]
    relations = [
        Relation(f"visible-{i}", edge["head_id"], edge["tail_id"], edge["relation"])
        for i, edge in enumerate(visible_edges)
    ]
    answer = Answer(
        text="、".join(item["name"] for item in question["gold_answers"]),
        subgraph=Subgraph(relations=relations),
        debug_info={"retrieval": {
            "gap_detected": True,
            "compensation_triggered": True,
            "compensation_documents": gap["compensation_gold_documents"],
            "temporary_relations": [{
                **masked,
                "supporting_documents": gap["compensation_gold_documents"],
            }],
        }},
    )
    metrics = BenchmarkScorer(payload).score(question, answer)
    assert metrics["retrieval"]["path_complete_rate"] < 1.0
    assert metrics["retrieval"]["recovered_path_complete_rate"] == 1.0
    assert metrics["gap"]["gap_detection_correct"] == 1.0
    assert metrics["gap"]["compensation_document_recall"] == 1.0


def test_oracle_control_records_query_specific_temporary_evidence():
    class Retriever:
        def retrieve(self, question, top_k=None, constraints=None):
            return RetrievalResult(
                retriever_name="hipporag2",
                chunks=[Chunk("graph", "graph-doc", "图证据", 0)],
            )

    class Index:
        def search(self, query, top_k):
            return [(Chunk("repair", "repair-doc", "补偿证据", 0), 0.9)]

    class Generator:
        def generate(self, question, result):
            return Answer(text="答案", citations=[Citation("S1", "repair-doc", "repair", 0, 4, "补偿证据")])

    method = BenchmarkHybridMethod.__new__(BenchmarkHybridMethod)
    method.oracle = True
    method.name = "oracle_repair"
    method.retriever = Retriever()
    method.index = Index()
    method.generator = Generator()
    method.default_top_k = 2
    constraints = RetrievalConstraints(
        (EdgeMask("actor", "acted_in", "film"),), ("演员 影片 出演",)
    )
    answer = method.ask("问题", top_k=2, constraints=constraints)
    debug = answer.debug_info["retrieval"]
    assert debug["compensation_documents"] == ["repair-doc"]
    assert debug["temporary_relations"][0]["supporting_documents"] == ["repair-doc"]
