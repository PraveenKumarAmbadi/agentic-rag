"""The API, agent, and eval modules must import without the ML stack.

Heavy libraries (torch, faiss, transformers, sentence-transformers) load only
when the stack is built, so these modules stay fast to import and the unit
tests can run without a GPU environment.
"""

import subprocess
import sys
import unittest

HEAVY = ("torch", "faiss", "transformers", "sentence_transformers")


class TestLightImports(unittest.TestCase):
    def test_no_heavy_modules_at_import(self):
        code = (
            "import sys, serving.main, agent.agent, agent.stack, evaluation.evaluate_rag, "
            "retrieval.fusion, scripts.build_indexes\n"
            f"print(','.join(m for m in {HEAVY!r} if m in sys.modules))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout.strip()
        self.assertEqual(out, "", f"heavy modules imported: {out}")


if __name__ == "__main__":
    unittest.main()
