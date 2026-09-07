# -*- coding: utf-8 -*-
"""finalize（人工复核闭环）核心逻辑单元测试：去重 / 补抽 / 校验确定性。"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import pandas as pd  # noqa: E402

from finalize import dedup_report, refill, final_check, norm, tokens, jaccard  # noqa: E402

MANIFEST = {
    "query": "query",
    "answer": "问答结果",
    "search_result": "搜索结果",
    "score": "score",
    "filename": "filename",
    "request_id": "request_id",
    "reference": "标注",
}


def make_cand(rows):
    return pd.DataFrame(rows, columns=["request_id", "query", "score", "问答结果", "搜索结果", "filename"])


class TestTextTools(unittest.TestCase):
    def test_norm_normalizes(self):
        self.assertEqual(norm("  Ceci   est-là "), "ceci est-là")

    def test_tokens_latin_cyrillic_arabic(self):
        self.assertIn("этот", tokens("Что означает этот пункт?"))
        self.assertIn("هذا", tokens("ما هو هذا المستند؟"))

    def test_jaccard(self):
        self.assertEqual(jaccard({"a", "b"}, {"a", "b"}), 1.0)
        self.assertEqual(jaccard({"a", "b"}, {"a", "c"}), 1 / 3)


class TestDedup(unittest.TestCase):
    def test_finds_duplicate_pair(self):
        df = make_cand([
            ("r1", "Quelle est la plus grande pyramide de l'Égypte antique ?", 0.8, "a", "c", "f1"),
            ("r2", "Quelle pyramide est la plus grande en Égypte antique ?", 0.8, "a", "c", "f2"),
            ("r3", "Quel est le rôle du glycogène ?", 0.8, "a", "c", "f3"),
        ])
        pairs = dedup_report(df, MANIFEST, th=0.3)
        self.assertEqual(len(pairs), 1)
        self.assertGreaterEqual(pairs[0][0], 0.7)
        self.assertEqual({pairs[0][3], pairs[0][4]}, {"r1", "r2"})

    def test_no_dup_for_unrelated(self):
        df = make_cand([
            ("r1", "Quel est le rôle du glycogène ?", 0.8, "a", "c", "f1"),
            ("r2", "Что такое социализация?", 0.8, "a", "c", "f2"),
        ])
        self.assertEqual(len(dedup_report(df, MANIFEST, th=0.3)), 0)


class TestRefill(unittest.TestCase):
    def setUp(self):
        self.cand = make_cand([
            ("keep1", "Какие базовые функции MPI входят в минимальный набор?", 0.8, "a", "c", "f"),
            ("keep2", "Что такое операционная система?", 0.8, "a", "c", "f"),
            ("p1", "Какова роль гликогена в животном организме?", 0.8, "a", "c", "f"),
            ("p2", "Какие функции выполняет ДНК в клетке?", 0.8, "a", "c", "f"),
            ("p3", "Какой размер шрифта нужен в отчёте?", 0.8, "a", "c", "f"),
            ("p4", "Какое сечение нейтрального провода принимают в трёхфазных сетях?", 0.8, "a", "c", "f"),
        ])

    def test_refill_excludes_kept(self):
        picked = refill(self.cand, MANIFEST, ["keep1", "keep2"], n=4, seed=42, th=0.3)
        ids = set(picked["request_id"])
        self.assertEqual(len(ids), 4)
        self.assertNotIn("keep1", ids)
        self.assertNotIn("keep2", ids)

    def test_refill_seed_reproducible(self):
        a = refill(self.cand, MANIFEST, ["keep1", "keep2"], n=4, seed=42, th=0.3)
        b = refill(self.cand, MANIFEST, ["keep1", "keep2"], n=4, seed=42, th=0.3)
        self.assertEqual(sorted(a["request_id"]), sorted(b["request_id"]))

    def test_refill_filters_similar_to_keep(self):
        # keep 里有一条问 "гликоген"，候选 p1 问同样内容 → 应被过滤
        cand = make_cand([
            ("keep1", "Какова роль гликогена в животном организме?", 0.8, "a", "c", "f"),
            ("p1", "Какова роль гликогена в животном?" , 0.8, "a", "c", "f"),
            ("p2", "Что такое ДНК?", 0.8, "a", "c", "f"),
        ])
        picked = refill(cand, MANIFEST, ["keep1"], n=1, seed=42, th=0.3)
        self.assertNotIn("p1", set(picked["request_id"]))
        self.assertIn("p2", set(picked["request_id"]))


class TestFinalCheck(unittest.TestCase):
    def _make_distinct(self, n=100):
        """构造两两 Jaccard < 0.3 的 n 条确定性 query（避免共享句式误报）。"""
        import random
        rng = random.Random(7)
        pool = [f"w{i:04d}" for i in range(4000)]
        rows = []
        for i in range(n):
            toks = rng.sample(pool, 8)
            rows.append((f"r{i}", " ".join(toks) + " ?", 0.8, "a", "c", "f"))
        return make_cand(rows)

    def test_check_passes(self):
        df = self._make_distinct()
        rep = final_check(df, MANIFEST)
        self.assertEqual(rep["n_unique"], 100)
        self.assertEqual(rep["problems"], [])

    def test_check_finds_count_mismatch(self):
        df = make_cand([(f"r{i}", f"Question {i}?", 0.8, "a", "c", "f") for i in range(99)])
        rep = final_check(df, MANIFEST)
        self.assertNotEqual(rep["n_unique"], 100)
        self.assertTrue(any("100" in p for p in rep["problems"]))

    def test_check_finds_dups(self):
        df = make_cand([
            ("r1", "Quelle est la plus grande pyramide de l'Égypte antique ?", 0.8, "a", "c", "f"),
            ("r2", "Quelle pyramide est la plus grande en Égypte antique ?", 0.8, "a", "c", "f"),
        ])
        rep = final_check(df, MANIFEST)
        self.assertTrue(any("重复对" in p for p in rep["problems"]))


if __name__ == "__main__":
    unittest.main()
