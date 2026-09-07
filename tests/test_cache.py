# -*- coding: utf-8 -*-
"""缓存续跑逻辑测试：判定结果追加写，重跑跳过已判定。"""
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import pandas as pd  # noqa: E402

from pipeline import _load_cache, _append_cache, classify_all  # noqa: E402
from fixtures.make_data import make_df  # noqa: E402

MANIFEST = {
    "query": "query",
    "answer": "问答结果",
    "search_result": "搜索结果",
    "score": "score",
    "filename": "filename",
    "request_id": "request_id",
    "reference": "标注",
}


class TestCacheResume(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache_path = os.path.join(self.tmp.name, "cache.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def test_append_and_load_roundtrip(self):
        rec = {"request_id": "r1", "is_specified": "是", "quality_issues": [], "reason": "x", "confidence": 0.9, "status": "ok"}
        _append_cache(self.cache_path, rec)
        _append_cache(self.cache_path, {"request_id": "r2", "status": "error", "error": "boom"})
        cache = _load_cache(self.cache_path)
        self.assertEqual(cache["r1"]["is_specified"], "是")
        self.assertEqual(cache["r2"]["status"], "error")

    def test_resume_skips_cached_ok(self):
        df = make_df([
            ("r1", "问题一?", 0.8, "回答一", "片段A"),
            ("r2", "问题二?", 0.7, "回答二", "片段B"),
            ("r3", "问题三?", 0.6, "回答三", "片段C"),
        ])
        # 预置 r1 已判定，且缓存中无 r2/r3
        _append_cache(self.cache_path, {"request_id": "r1", "is_specified": "否", "quality_issues": [], "reason": "ok", "confidence": 1.0, "status": "ok"})

        calls = []

        def fake_classify_batch(batch_items, language_hint, model, base_url, api_key):
            out = []
            for rid, q, a, nf in batch_items:
                calls.append(rid)
                out.append((rid, {"request_id": rid, "is_specified": "否", "quality_issues": [], "reason": "ok", "confidence": 1.0, "status": "ok"}))
            return out

        with patch("pipeline.classify_batch", side_effect=fake_classify_batch):
            result = classify_all(df, MANIFEST, "model", "http://x", "key", self.cache_path, workers=3)

        self.assertEqual(sorted(calls), ["r2", "r3"])  # r1 命中缓存不重判
        self.assertEqual(len(result), 3)
        self.assertEqual(result["r1"]["is_specified"], "否")

    def test_error_records_retried(self):
        df = make_df([("r1", "问题一?", 0.8, "回答一", "片段A")])
        # 预置 r1 失败记录
        _append_cache(self.cache_path, {"request_id": "r1", "status": "error", "error": "boom"})

        calls = []

        def fake_classify_batch(batch_items, language_hint, model, base_url, api_key):
            out = []
            for rid, q, a, nf in batch_items:
                calls.append(rid)
                out.append((rid, {"request_id": rid, "is_specified": "否", "quality_issues": [], "reason": "ok", "confidence": 1.0, "status": "ok"}))
            return out

        with patch("pipeline.classify_batch", side_effect=fake_classify_batch):
            classify_all(df, MANIFEST, "model", "http://x", "key", self.cache_path, workers=1)
        self.assertEqual(calls, ["r1"])  # error 状态会被重试


if __name__ == "__main__":
    unittest.main()

