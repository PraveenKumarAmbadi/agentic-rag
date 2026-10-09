"""Citation parsing and the single refusal definition."""

import unittest

from agent.citations import (
    build_citation_map,
    classify_refusal_prefix,
    is_refusal,
    parse_citations,
    refusal_reason,
)


class TestRefusalDetection(unittest.TestCase):
    def test_marker_variants(self):
        for text in (
            "INSUFFICIENT_EVIDENCE: no data",
            "INSUFFICIENT EVIDENCE: no data",
            "insufficient evidence - no data",
            "**INSUFFICIENT_EVIDENCE**: no data",
            "\n  Insufficient_Evidence.",
            "INSUFFICIENT_EVIDENCE",
        ):
            with self.subTest(text=text):
                self.assertTrue(is_refusal(text))

    def test_non_refusals(self):
        for text in (
            "The effect was significant [1].",
            "Insufficient data was reported [2].",
            "INSUFFICIENT EVIDENCED claims",
            "",
        ):
            with self.subTest(text=text):
                self.assertFalse(is_refusal(text))

    def test_prefix_classification_while_streaming(self):
        self.assertEqual(classify_refusal_prefix(""), "undecided")
        self.assertEqual(classify_refusal_prefix("**INSUFF"), "undecided")
        self.assertEqual(classify_refusal_prefix("INSUFFICIENT EVIDENCE"), "undecided")
        self.assertEqual(classify_refusal_prefix("INSUFFICIENT EVIDENCE:"), "refusal")
        self.assertEqual(classify_refusal_prefix("The"), "answer")
        self.assertEqual(classify_refusal_prefix("INSUFFICIENT EVIDENCE", final=True), "refusal")
        self.assertEqual(classify_refusal_prefix("INSUFF", final=True), "answer")

    def test_reason_extraction(self):
        self.assertEqual(refusal_reason("INSUFFICIENT_EVIDENCE: no RCT on X"), "no RCT on X")
        self.assertEqual(refusal_reason("**Insufficient evidence** - nothing\non X"), "nothing on X")
        self.assertEqual(refusal_reason("INSUFFICIENT_EVIDENCE"), "")
        self.assertEqual(refusal_reason("An answer [1]."), "")


class TestCitations(unittest.TestCase):
    def test_valid_invalid_and_grounded(self):
        parsed = parse_citations("A [1]. B [3][1]. C [12].", num_docs=3)
        self.assertEqual(parsed, {"markers": [1, 3], "invalid": [12], "grounded": False})
        self.assertTrue(parse_citations("A [2].", 3)["grounded"])
        self.assertFalse(parse_citations("No citations.", 3)["grounded"])

    def test_citation_map(self):
        self.assertEqual(build_citation_map([1, 3], ["a", "b", "c"]), {"1": "a", "3": "c"})


if __name__ == "__main__":
    unittest.main()
