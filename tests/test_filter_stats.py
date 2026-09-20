# -*- coding: utf-8 -*-
"""P2.6: filter_stats visibility + prefer-uncertain policy documentation."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from pipeline import apply_filter, summarize_filter_stats  # noqa: E402
import pandas as pd  # noqa: E402

MANIFEST = {
    "query": "query",
    "answer": "问答结果",
    "search_result": "搜索结果",
    "score": "score",
    "filename": "filename",
    "request_id": "request_id",
    "reference": "标注",
}


class TestFilterStats(unittest.TestCase):
    def test_summarize_counts_and_difficulty_shift(self):
        removed = {
            "r1": ["指定query"],
            "r2": ["灰色待裁决"],
            "r3": ["判定失败: timeout"],
            "r4": ["质量:empty_answer", "chunk全空"],
        }
        stats = summarize_filter_stats(removed, n_input=10)
        self.assertEqual(stats["n_input"], 10)
        self.assertEqual(stats["n_removed"], 4)
        self.assertEqual(stats["n_kept"], 6)
        self.assertEqual(stats["reason_counts"]["specified_query"], 1)
        self.assertEqual(stats["reason_counts"]["gray_pending"], 1)
        self.assertEqual(stats["reason_counts"]["classify_failure"], 1)
        self.assertEqual(stats["reason_counts"]["quality:empty_answer"], 1)
        self.assertEqual(stats["reason_counts"]["chunk_empty"], 1)
        self.assertEqual(stats["difficulty_shift"]["tie_break_policy"], "prefer_uncertain")
        self.assertIn("存疑", stats["difficulty_shift"]["note"])

    def test_apply_filter_still_fail_closed_on_gray(self):
        df = pd.DataFrame([
            {"request_id": "r1", "query": "q", "问答结果": "a", "搜索结果": "c", "score": 1, "filename": "f", "标注": ""},
        ])
        cand, removed = apply_filter(df, MANIFEST, {"r1": {"status": "ok", "is_specified": "存疑", "quality_issues": []}})
        self.assertIn("r1", removed)
        self.assertIn("灰色待裁决", removed["r1"])
        self.assertTrue(cand.empty)


if __name__ == "__main__":
    unittest.main()
