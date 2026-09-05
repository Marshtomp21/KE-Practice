"""旧题集移除后，前端仍有开发集示例且不暴露评测答案。"""
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from src.api.server import app


def test_examples_use_only_benchmark_dev_questions():
    root = Path(__file__).resolve().parents[1]
    payload = yaml.safe_load(
        (root / "eval/benchmark_v2/questions.yaml").read_text(encoding="utf-8")
    )
    dev = {q["question"] for q in payload["questions"] if q["split"] == "dev"}
    with TestClient(app) as client:
        response = client.get("/api/examples")
        assert response.status_code == 200
        examples = response.json()["examples"]
        assert examples
        assert len(examples) <= 6
        assert len({q["kind"] for q in examples}) == len(examples)
        for item in examples:
            assert item["question"] in dev
            assert set(item) == {"kind", "label", "question"}
            assert item["label"] != item["kind"]
        assert len(client.get("/api/examples?limit=1").json()["examples"]) == 1
        assert client.get("/api/examples?limit=0").json() == {"examples": []}


def test_health_keeps_five_methods_without_legacy_graph_snapshot():
    with TestClient(app) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["ready"] is True
        assert {"vector", "kg2rag", "hipporag2", "gap_repair", "library_graphrag"} <= set(payload["retrievers"])
        assert "dataset_graph_source" in payload["graph"]
        assert "local_graph_snapshot" not in payload["graph"]
