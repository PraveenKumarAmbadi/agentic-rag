"""search_hybrid degradation, breakers, and RRF — retrievers are patched."""

import logging
import unittest
from unittest import mock

from retrieval import fusion
from retrieval.types import SearchResult
from tracing import CircuitBreaker

logging.disable(logging.CRITICAL)


def results(*ids):
    return [SearchResult(doc_id=i, score=1.0, title=i, text=i) for i in ids]


def run(**kwargs):
    return fusion.search_hybrid(
        query_text="q", query_id="t", bm25_client=None, splade_client=None,
        faiss_index=None, corpus={}, top_k=10, **kwargs,
    )


class TestSearchHybrid(unittest.TestCase):
    def setUp(self):
        for name, ids in (("search_bm25", ("a", "b")), ("search_splade", ("b", "c")), ("search_dense", ("b",))):
            patcher = mock.patch.object(fusion, name, return_value=results(*ids))
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_fuses_all_methods(self):
        out = run()
        self.assertEqual(out.failed_methods, ())
        self.assertEqual(out.results[0].doc_id, "b")  # found by all three

    def test_failed_method_is_reported_and_skipped(self):
        fusion.search_bm25.side_effect = ConnectionError("es down")
        out = run()
        self.assertEqual(out.failed_methods, ("bm25",))
        self.assertEqual({r.doc_id for r in out.results}, {"b", "c"})

    def test_open_breaker_skips_the_call(self):
        breaker = CircuitBreaker("bm25", failure_threshold=1, cooldown_s=60)
        fusion.search_bm25.side_effect = ConnectionError("es down")
        run(bm25_breaker=breaker)
        fusion.search_bm25.reset_mock()
        out = run(bm25_breaker=breaker)
        fusion.search_bm25.assert_not_called()
        self.assertEqual(out.failed_methods, ("bm25",))

    def test_query_text_is_not_logged(self):
        previous = logging.root.manager.disable
        logging.disable(logging.NOTSET)
        self.addCleanup(logging.disable, previous)
        with self.assertLogs("retrieval.fusion", level="INFO") as logs:
            fusion.search_hybrid(
                query_text="secret patient question", query_id="t", bm25_client=None,
                splade_client=None, faiss_index=None, corpus={},
            )
        self.assertNotIn("secret patient question", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
