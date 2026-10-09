"""Index builder: completeness checks decide skip vs. rebuild."""

import json
import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts import build_indexes as bi

logging.disable(logging.CRITICAL)


class FakeIndices:
    def __init__(self, existing):
        self.existing = dict(existing)  # name -> doc count
        self.deleted, self.created = [], []

    def exists(self, index):
        return index in self.existing

    def refresh(self, index):
        pass

    def delete(self, index):
        self.deleted.append(index)
        self.existing.pop(index)

    def create(self, index, mappings):
        self.created.append(index)
        self.existing[index] = 0


class FakeES:
    def __init__(self, existing=()):
        self.indices = FakeIndices(existing)

    def count(self, index):
        return {"count": self.indices.existing[index]}


def docs(n):
    return [SimpleNamespace(doc_id=str(i), title=f"t{i}", text=f"x{i}", full_text=f"t{i} x{i}") for i in range(n)]


CFG = {"bm25": {"index_name": "bm25"}, "splade": {"index_name": "splade"}}


class TestEsCompleteness(unittest.TestCase):
    def test_complete_index_is_skipped(self):
        es = FakeES({"bm25": 3})
        self.assertEqual(bi.build_bm25(es, CFG, docs(3), recreate=False), "present")
        self.assertEqual(es.indices.deleted, [])

    def test_missing_or_partial_index_is_rebuilt(self):
        for existing in ({}, {"bm25": 2}):
            with self.subTest(existing=existing):
                self.assertFalse(bi._es_index_complete(FakeES(existing), "bm25", 3))

    def test_recreate_forces_rebuild(self):
        es = FakeES({"bm25": 3})
        rebuilt = []
        original = bi._rebuild_es_index
        bi._rebuild_es_index = lambda es, name, mappings, actions, expected: rebuilt.append((name, len(list(actions))))
        try:
            self.assertEqual(bi.build_bm25(es, CFG, docs(3), recreate=True), "built")
        finally:
            bi._rebuild_es_index = original
        self.assertEqual(rebuilt, [("bm25", 3)])


class TestDenseCompleteness(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.index = Path(tmp.name) / "d.faiss"
        self.ids = Path(tmp.name) / "d_ids.json"

    def test_matching_ids_are_complete(self):
        self.index.write_bytes(b"x")
        self.ids.write_text(json.dumps(["1", "2"]))
        self.assertTrue(bi._dense_index_complete(self.index, self.ids, {"1", "2"}))

    def test_mismatch_missing_or_corrupt_is_incomplete(self):
        self.assertFalse(bi._dense_index_complete(self.index, self.ids, {"1"}))  # missing
        self.index.write_bytes(b"x")
        self.ids.write_text(json.dumps(["1", "3"]))
        self.assertFalse(bi._dense_index_complete(self.index, self.ids, {"1", "2"}))
        self.ids.write_text("{not json")
        self.assertFalse(bi._dense_index_complete(self.index, self.ids, {"1", "2"}))


if __name__ == "__main__":
    unittest.main()
