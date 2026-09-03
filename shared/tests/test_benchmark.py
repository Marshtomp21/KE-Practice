from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

SHARED = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHARED / "scripts"))

from benchmark_utils import normalize_text, score_answer


class ScorerTests(unittest.TestCase):
    def entity(self, canonical_id="Q1856626", name="陆川 (导演)", aliases=None):
        return {"canonical_id": canonical_id, "type": "Person", "name": name, "aliases": aliases or [name, "陆川"]}

    def test_alias_is_accepted(self):
        entity = self.entity()
        question = {"answerable": True, "answer_kind": "entity", "gold_answers": [entity], "answer_candidates": [entity]}
        self.assertEqual(score_answer(question, "导演是陆川。") ["score"], 1.0)

    def test_refusal_with_gold_name_is_not_false_positive(self):
        entity = self.entity(name="汉娜·沃丁厄姆", aliases=["汉娜·沃丁厄姆"])
        question = {"answerable": True, "answer_kind": "entity_set", "gold_answers": [entity], "answer_candidates": [entity]}
        result = score_answer(question, "无法回答。资料只提到了汉娜·沃丁厄姆，但没有跨影片证据。")
        self.assertEqual(result["score"], 0.0)
        self.assertTrue(result["abstained"])

    def test_entity_set_uses_precision_recall_f1(self):
        a = self.entity("A", "甲", ["甲"])
        b = self.entity("B", "乙", ["乙"])
        c = self.entity("C", "丙", ["丙"])
        question = {"answerable": True, "answer_kind": "entity_set", "gold_answers": [a, b], "answer_candidates": [a, b, c]}
        result = score_answer(question, "甲和丙。")
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 0.5)
        self.assertEqual(result["f1"], 0.5)

    def test_entity_set_does_not_count_explanatory_candidates(self):
        a = self.entity("A", "甲", ["甲"])
        b = self.entity("B", "乙", ["乙"])
        c = self.entity("C", "丙", ["丙"])
        question = {"answerable": True, "answer_kind": "entity_set", "gold_answers": [a, b], "answer_candidates": [a, b, c]}
        answer = "答案如下：\n\n- 甲\n- 乙\n\n证据对比还提到了丙，但丙只出现一次。"
        result = score_answer(question, answer)
        self.assertEqual(result["score"], 1.0)
        self.assertIn("bullet block", result["reason"])

    def test_labelled_conclusion_has_priority_over_evidence_list(self):
        a = self.entity("A", "甲", ["甲"])
        b = self.entity("B", "乙", ["乙"])
        question = {"answerable": True, "answer_kind": "entity_set", "gold_answers": [a], "answer_candidates": [a, b]}
        answer = "- 影片一：甲、乙\n- 影片二：甲\n\n**结论**：只有甲重复出演。"
        result = score_answer(question, answer)
        self.assertEqual(result["score"], 1.0)
        self.assertIn("labelled conclusion", result["reason"])

    def test_explicit_summary_has_priority_over_evidence_bullets(self):
        a = self.entity("A", "甲", ["甲"])
        b = self.entity("B", "乙", ["乙"])
        question = {"answerable": True, "answer_kind": "entity_set", "gold_answers": [a], "answer_candidates": [a, b]}
        answer = "### 甲\n- 影片一\n- 影片二\n\n以上信息显示，甲重复出演。"
        result = score_answer(question, answer)
        self.assertEqual(result["score"], 1.0)
        self.assertIn("summary sentence", result["reason"])

    def test_markdown_summary_heading_has_priority_over_evidence_bullets(self):
        a = self.entity("A", "甲", ["甲"])
        b = self.entity("B", "乙", ["乙"])
        question = {"answerable": True, "answer_kind": "entity_set", "gold_answers": [a], "answer_candidates": [a, b]}
        answer = "- 乙仅有一部证据\n\n### 总结\n\n只有甲满足条件。"
        result = score_answer(question, answer)
        self.assertEqual(result["score"], 1.0)
        self.assertIn("labelled conclusion", result["reason"])

    def test_no_answer_requires_explicit_abstention(self):
        question = {"answerable": False, "answer_kind": "abstention", "gold_answers": [], "answer_candidates": []}
        self.assertEqual(score_answer(question, "无法根据现有资料确定。") ["score"], 1.0)
        self.assertEqual(score_answer(question, "无法从提供的资料中得知。") ["score"], 1.0)
        self.assertEqual(score_answer(question, "The provided text does not contain information about it.") ["score"], 1.0)
        self.assertEqual(score_answer(question, "I am unable to answer this question given the provided data.") ["score"], 1.0)
        self.assertEqual(score_answer(question, "导演是某某。") ["score"], 0.0)


class FrozenBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = SHARED / "benchmarks" / "l2_film_120_v2"
        cls.manifest = [json.loads(line) for line in (cls.root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
        cls.questions = {
            split: [json.loads(line) for line in (cls.root / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
            for split in ("dev", "test")
        }

    def test_expected_split_sizes(self):
        self.assertEqual(len(self.manifest), 120)
        self.assertEqual(len(self.questions["dev"]), 5)
        self.assertEqual(len(self.questions["test"]), 20)

    def test_every_evidence_span_matches_document(self):
        by_id = {row["doc_id"]: (self.root / row["path"]).read_text(encoding="utf-8") for row in self.manifest}
        for questions in self.questions.values():
            for question in questions:
                for evidence in question["gold_evidence"]:
                    text = by_id[evidence["doc_id"]]
                    self.assertEqual(text[evidence["char_start"]:evidence["char_end"]], evidence["quote"])

    def test_aggregate_answers_require_two_distinct_films(self):
        for questions in self.questions.values():
            for question in questions:
                if question["answer_kind"] != "entity_set":
                    continue
                for answer_path in question["required_relation_paths"]:
                    films = {path["nodes"][1] for path in answer_path["distinct_film_paths"]}
                    self.assertGreaterEqual(len(films), 2)

    def test_candidates_have_clean_unique_aliases(self):
        for questions in self.questions.values():
            for question in questions:
                owners = {}
                for entity in question["answer_candidates"]:
                    self.assertNotIn("None", entity["aliases"], question["id"])
                    self.assertNotRegex(entity["name"], r"[{}\[\]|]", question["id"])
                    for alias in entity["aliases"]:
                        key = normalize_text(alias)
                        previous = owners.setdefault(key, entity["canonical_id"])
                        self.assertEqual(previous, entity["canonical_id"], f"{question['id']}: ambiguous alias {alias}")

    def test_manually_repaired_duplicate_cast_entries(self):
        fight_club = (self.root / "documents" / "film_Q190050.txt").read_text(encoding="utf-8")
        movie_2046 = (self.root / "documents" / "film_Q164702.txt").read_text(encoding="utf-8")
        self.assertNotIn("布莱德·彼特", fight_club)
        self.assertEqual(movie_2046.count("张震"), 1)


if __name__ == "__main__":
    unittest.main()
