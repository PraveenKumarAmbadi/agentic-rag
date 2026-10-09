"""Integrity of the frozen eval sets in data/eval/ (no corpus needed)."""

import json
import unittest
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "eval"


def load(name):
    with (EVAL_DIR / name).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class TestEvalSets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.answerable = load("answerable.jsonl")
        cls.unanswerable = load("unanswerable.jsonl")
        cls.adversarial = load("adversarial.jsonl")

    def test_ids_present_and_unique_across_sets(self):
        ids = [r["query_id"] for s in (self.answerable, self.unanswerable, self.adversarial) for r in s]
        self.assertTrue(all(ids))
        self.assertEqual(len(ids), len(set(ids)))

    def test_questions_unique_and_non_empty(self):
        qs = [r["question"].strip().lower() for s in (self.answerable, self.unanswerable, self.adversarial) for r in s]
        self.assertTrue(all(qs))
        self.assertEqual(len(qs), len(set(qs)))

    def test_set_labels_and_gold(self):
        for r in self.answerable:
            self.assertEqual(r["set"], "answerable")
            self.assertTrue(r["gold_doc_ids"])
        for r in self.unanswerable:
            self.assertEqual(r["set"], "unanswerable")
            self.assertEqual(r["gold_doc_ids"], [])

    def test_adversarial_items_are_well_formed(self):
        answerable_ids = {r["query_id"] for r in self.answerable}
        for r in self.adversarial:
            with self.subTest(r["query_id"]):
                self.assertEqual(r["set"], "adversarial")
                self.assertEqual(r["gold_doc_ids"], [])           # nothing answers it
                self.assertTrue(r["near_miss_doc_ids"])            # but something looks like it does
                self.assertTrue(r["trap"] and r["rationale"])
                self.assertIn(r["paired_query_id"], answerable_ids)


if __name__ == "__main__":
    unittest.main()
