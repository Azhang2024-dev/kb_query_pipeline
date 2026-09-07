# -*- coding: utf-8 -*-
"""pipeline 核心逻辑单元测试：filter / sample / convert 确定性断言。"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import pandas as pd  # noqa: E402

from pipeline import apply_filter, stratified_sample, to_zhfmt  # noqa: E402
from fixtures.make_data import make_df, make_golden_df  # noqa: E402

MANIFEST = {
    "query": "query",
    "answer": "问答结果",
    "search_result": "搜索结果",
    "score": "score",
    "filename": "filename",
    "request_id": "request_id",
    "reference": "标注",
}


class TestFilter(unittest.TestCase):
    def setUp(self):
        self.df = make_df([
            ("r1", "正常问题一?", 0.8, "回答一", "片段A"),
            ("r1", "正常问题一?", 0.6, "回答一", "片段B"),
            ("r2", "这个合同有效期多久?", 0.9, "回答二", "片段C"),
            ("r3", "Что означает этот пункт договора?", 0.7, "回答三", "片段D"),
            ("r4", "空回答问题?", 0.8, "", "片段E"),
            ("r5", "chunk全空?", 0.8, "回答五", ""),
        ])

    def test_specified_removed(self):
        verdicts = {
            "r1": {"status": "ok", "is_specified": "否", "quality_issues": []},
            "r2": {"status": "ok", "is_specified": "是", "quality_issues": []},
            "r3": {"status": "ok", "is_specified": "是", "quality_issues": []},
            "r4": {"status": "ok", "is_specified": "否", "quality_issues": []},
            "r5": {"status": "ok", "is_specified": "否", "quality_issues": []},
        }
        cand, removed = apply_filter(self.df, MANIFEST, verdicts)
        kept = set(cand["request_id"].astype(str).unique())
        self.assertNotIn("r2", kept)
        self.assertIn("r2", removed)
        self.assertIn("r3", removed)
        self.assertEqual(len(removed), 3)  # r2, r3 指定 + r5 chunk全空
        self.assertIn("指定query", removed["r2"])
        self.assertIn("指定query", removed["r3"])

    def test_quality_removed(self):
        verdicts = {
            "r1": {"status": "ok", "is_specified": "否", "quality_issues": []},
            "r4": {"status": "ok", "is_specified": "否", "quality_issues": ["empty_answer"]},
            "r5": {"status": "ok", "is_specified": "否", "quality_issues": []},
        }
        cand, removed = apply_filter(self.df, MANIFEST, verdicts)
        self.assertIn("r4", removed)
        self.assertIn("质量:empty_answer", removed["r4"])

    def test_chunk_empty_fallback(self):
        verdicts = {
            "r1": {"status": "ok", "is_specified": "否", "quality_issues": []},
            "r5": {"status": "ok", "is_specified": "否", "quality_issues": []},
        }
        cand, removed = apply_filter(self.df, MANIFEST, verdicts)
        self.assertIn("r5", removed)
        self.assertIn("chunk全空", removed["r5"])

    def test_gray_pending_removed(self):
        verdicts = {
            "r1": {"status": "ok", "is_specified": "存疑", "quality_issues": []},
        }
        cand, removed = apply_filter(self.df, MANIFEST, verdicts)
        self.assertIn("r1", removed)
        self.assertIn("灰色待裁决", removed["r1"])

    def test_gray_verdict_keep(self):
        verdicts = {"r1": {"status": "ok", "is_specified": "存疑", "quality_issues": []}}
        cand, removed = apply_filter(self.df, MANIFEST, verdicts, gray_verdicts={"r1": "保留"})
        self.assertNotIn("r1", removed)


class TestSample(unittest.TestCase):
    def test_seed_reproducible(self):
        ids = [f"rid_{i}" for i in range(20)]
        rows = {rid: 1 + (len(rid) % 6) for rid in ids}
        a = stratified_sample(ids, rows, [(1, 2), (3, 5), (6, 10**9)], 42, 10)
        b = stratified_sample(ids, rows, [(1, 2), (3, 5), (6, 10**9)], 42, 10)
        self.assertEqual(sorted(a), sorted(b))
        self.assertEqual(len(a), 10)
        self.assertEqual(len(set(a)), 10)

    def test_seed_changes_result(self):
        ids = [f"rid_{i}" for i in range(20)]
        rows = {rid: 1 + (len(rid) % 6) for rid in ids}
        a = stratified_sample(ids, rows, [(1, 2), (3, 5), (6, 10**9)], 42, 10)
        c = stratified_sample(ids, rows, [(1, 2), (3, 5), (6, 10**9)], 7, 10)
        self.assertNotEqual(sorted(a), sorted(c))

    def test_quota_by_stratum(self):
        ids = [f"rid_{i}" for i in range(30)]
        rows = {}
        for i, rid in enumerate(ids):
            rows[rid] = 1 if i < 10 else (3 if i < 20 else 7)
        sel = stratified_sample(ids, rows, [(1, 2), (3, 5), (6, 10**9)], 42, 15)
        from collections import Counter
        strata = Counter()
        for rid in sel:
            n = rows[rid]
            strata[next(lo for lo, hi in [(1, 2), (3, 5), (6, 10**9)] if lo <= n <= hi)] += 1
        # 30 条 1-2 层 10 条 → 配额 5；3-5 层 10 条 → 5；6+ 层 10 条 → 5
        self.assertEqual(strata[1], 5)
        self.assertEqual(strata[3], 5)
        self.assertEqual(strata[6], 5)


class TestConvert(unittest.TestCase):
    def test_block_format(self):
        df = make_df([
            ("r1", "问题一?", 0.5, "回答一", "片段A"),
            ("r1", "问题一?", 0.9, "回答一", "片段B"),
            ("r2", "问题二?", 0.7, "回答二", "片段C"),
        ])
        zh = to_zhfmt(df, MANIFEST)
        self.assertEqual(list(zh.columns), ["query", "标注", "filename", "搜索结果", "score", "大模型问答结果"])
        # 块首行：query/回答有值；块内 score 降序
        self.assertEqual(zh["query"].notna().sum(), 2)
        self.assertEqual(zh["大模型问答结果"].notna().sum(), 2)
        blk = zh.iloc[0:2]
        self.assertGreater(blk["score"].iloc[0], blk["score"].iloc[1])
        self.assertTrue(pd.isna(zh["query"].iloc[1]))


class TestGoldenPrompts(unittest.TestCase):
    """黄金用例：判定 prompt 必须覆盖严格口径（指示词 + 具名实体指向）。"""

    def test_prompt_mentions_inflection_cases(self):
        from prompts import CLASSIFY_RULES
        for token in ["этой теме", "этого сборника", "у того", "этим жилым помещением"]:
            self.assertIn(token, CLASSIFY_RULES)

    def test_prompt_excludes_time_and_definition(self):
        from prompts import CLASSIFY_RULES
        for token in ["tahun ini", "X itu apa", "персональных данных"]:
            self.assertIn(token, CLASSIFY_RULES)

    def test_prompt_arabic_aforementioned(self):
        from prompts import CLASSIFY_RULES
        self.assertIn("المذكور", CLASSIFY_RULES)

    def test_prompt_russian_determiners(self):
        from prompts import CLASSIFY_RULES
        for token in ["данный", "настоящий", "указанный"]:
            self.assertIn(token, CLASSIFY_RULES)

    def test_prompt_demonstrative_rule(self):
        from prompts import CLASSIFY_RULES
        for token in ["ce projet", "this/that", "ЭТОТ/эта"]:
            self.assertIn(token.lower(), CLASSIFY_RULES.lower())

    def test_prompt_named_entity_strict(self):
        """严格口径：具名实体指向必须判指定。"""
        from prompts import CLASSIFY_RULES
        for token in ["Sarah Estelle Touvoli", "Gamba-Ivinga", "Traccar", "baccalauréat général 2024",
                      "код 31", "نموذج توصية طالب الماستر", "frite vendue à 1000", "كتاب كفاية الأصول"]:
            self.assertIn(token, CLASSIFY_RULES)

    def test_prompt_named_institution_exception(self):
        """例外：具名机构但问通用规则 → 非指定。"""
        from prompts import CLASSIFY_RULES
        self.assertIn("Ambassade de Chine au Togo", CLASSIFY_RULES)

    def test_prompt_language_context(self):
        from prompts import CLASSIFY_PROMPT
        self.assertIn("{language_hint}", CLASSIFY_PROMPT)

    def test_batch_prompt_exists(self):
        from prompts import CLASSIFY_BATCH_PROMPT
        self.assertIn("{questions}", CLASSIFY_BATCH_PROMPT)
        self.assertIn("CLASSIFY_RULES" if False else "results", CLASSIFY_BATCH_PROMPT)

    def test_golden_constants_updated(self):
        from pipeline import GOLDEN_SPECIFIED, GOLDEN_NOT_SPECIFIED
        joined = "\n".join(GOLDEN_SPECIFIED + GOLDEN_NOT_SPECIFIED)
        self.assertIn("Sarah Estelle Touvoli", joined)
        self.assertIn("Ambassade de Chine au Togo", joined)


if __name__ == "__main__":
    unittest.main()
