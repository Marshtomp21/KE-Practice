from eval.prepare_expanded import batch_key
import copy
import sys

import pytest
import yaml

from eval import run_gap_repair
from src.core.config import load_settings


def test_embedding_checkpoint_is_bound_to_text_order_and_encoder():
    encoder = {"model": "bge-m3", "dimension": 1024}
    key = batch_key(["title\ntext", "other"], encoder)
    assert key == batch_key(["title\ntext", "other"], dict(encoder))
    assert key != batch_key(["other", "title\ntext"], encoder)
    assert key != batch_key(["title\nchanged", "other"], encoder)
    assert key != batch_key(["title\ntext", "other"], {**encoder, "model": "different"})


def test_runner_rejects_wrong_expanded_gold_before_loading_index(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("films.jsonl", "actors.jsonl", "directors.jsonl", "relations.jsonl"):
        (source / name).write_text("", encoding="utf-8")
    config = copy.deepcopy(load_settings().as_dict())
    config["paths"]["dataset_dir"] = str(source)
    settings = tmp_path / "settings.yaml"
    settings.write_text(yaml.safe_dump(config), encoding="utf-8")
    questions = tmp_path / "questions.yaml"
    questions.write_text(yaml.safe_dump({"benchmark": {"source_sha256": "wrong", "question_count": 1},
                                         "questions": [{"id": "fixture"}], "entity_catalog": {}}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["run_gap_repair", "--settings", str(settings),
                                    "--questions", str(questions)])
    with pytest.raises(SystemExit, match="Source digest mismatch"):
        run_gap_repair.main()


def test_runner_does_not_mislabel_deterministic_baseline(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_gap_repair", "--method", "vector"])
    with pytest.raises(SystemExit) as exc:
        run_gap_repair.main()
    assert exc.value.code == 2
