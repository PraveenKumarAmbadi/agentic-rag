"""SPLADE sparsification (needs torch; skipped in the core-deps CI job)."""

import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestSparsify(unittest.TestCase):
    def test_weights_that_round_to_zero_are_dropped(self):
        import torch

        from indexing.splade_encoder import SpladeEncoder

        # 3e-7 is positive but rounds to 0.0 at 6 decimals; Elasticsearch
        # rejects 0.0 as a sparse_vector feature value.
        vectors = torch.tensor([[1.5, 3e-7, 0.0, 2e-6]])
        (doc,) = SpladeEncoder._sparsify(vectors, top_k=4)
        self.assertEqual(doc, {0: 1.5, 3: 2e-6})
        self.assertTrue(all(w > 0.0 for w in doc.values()))


if __name__ == "__main__":
    unittest.main()
