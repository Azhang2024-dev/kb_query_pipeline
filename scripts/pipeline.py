# -*- coding: utf-8 -*-
"""kb-query-pipeline 主入口：认列 → 判定 → 剔除 → 分层抽样 → 转 zhfmt。

五阶段，单入口 CLI：
  1. 认列（LLM 1 次）：表头 + 样本值 → 列语义映射 manifest（打印确认）
  2. 判定（LLM 批量并发）：逐批（默认 20 条/批）判 指定/质量多维，缓存续跑
  3. 剔除（确定性）：应用判定 + chunk 全空兜底 → 候选集 + 剔除清单
  4. 分层抽样（确定性）：按召回行数分层配额随机抽样 N，seed 固定
  5. 转 zhfmt（确定性）：长格式 → 6 列块状

人工复核闭环（抽样后标黄 → 差额补抽 → 最终校验）见 scripts/finalize.py。

API 配置（环境变量）：KB_QA_API_KEY（必填）、KB_QA_API_BASE_URL（可选）、KB_QA_MODEL（可选）。
换模型后需删除 *.classify_cache.jsonl 重新判定。
"""
import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

sys.stdout.reconfigure(encoding="utf-8")

try:
    import pandas as pd
except ImportError:
    pd = None

from prompts import CLASSIFY_PROMPT, CLASSIFY_BATCH_PROMPT, SCHEMA_PROMPT

DEFAULT_BASE_URL = "https://maragate.aibotplatform.com/v1"
API_KEY_ENV = "KB_QA_API_KEY"
BASE_URL_ENV = "KB_QA_API_BASE_URL"

# zhfmt 输出列顺序
ZH_COLS = ["query", "标注", "filename", "搜索结果", "score", "大模型问答结果"]

# 黄金回归用例：必须判为"指定" / 必须判为"非指定"
GOLDEN_SPECIFIED = [
    # 指示词指向（原有）
    "Сколько длится лекционное занятие по этой теме?",
    "Какой УИН указан для перечисления задолженности по этому взысканию?",
    "Какова сумма комиссии по этой безналичной оплате услуг?",
    "Какова главная особенность этого сборника зелий?",
    "От чего зависят свойства каждого зелья в этом наборе?",
    # 具名实体指向（新增严格口径，来自实战否决）
    "Comment contacter Sarah Estelle Touvoli par e-mail ?",
    "Quel est le pourcentage global de fermeture des actions HAZOP sur les sites de Gamba-Ivinga et Atora ?",
    "Combien de sujets le candidat peut-il choisir pour l'épreuve de philosophie du baccalauréat général 2024 ?",
    "Quel est le format du SMS de configuration pour connecter une montre en protocole watch à un serveur Traccar ?",
    "Как расшифровывается иное код 31 в больничном листе?",
    "ما الخطوات اللازمة لاستكمال نموذج توصية طالب الماستر وتوقيعه وختمه؟",
    "Quel est le poids de la portion de frite vendue à 1000 ?",
    "كيف ينقسم محتوى كتاب كفاية الأصول إلى مباحث الألفاظ والأصول العملية والعقلية؟",
    "Quel est le thème du stage de recherche effectué du 02 décembre au 18 janvier 2024 ?",
]
GOLDEN_NOT_SPECIFIED = [
    "Как подать документы в Академию через Госуслуги?",
    "Quels sont les trois critères d'une bonne formulation d'un sujet de recherche ?",
    "Какой объем персональных данных можно передавать третьим лицам для цели обработки?",
    # 通用规则/通用知识（新增，来自实战认可）
    "À quels étudiants la bourse d'encouragement de l'Ambassade de Chine au Togo est-elle destinée ?",
    "Quel type de régime politique a la Côte d’Ivoire ?",
    "Comment expliquer le concept de l’économie circulaire ?",
    "Какое сечение нейтрального провода принимают в трехфазных четырехпроводных сетях?",
    "كيف تصبح الترجمة الناتجة عن الحاسوب صالحة للنشر؟",
    "هل يجب إزالة القفازات عند مصافحة الآخرين؟",
]


def _require_pd():
    if pd is None:
        raise RuntimeError("pandas 未安装；请使用 kb-qa-eval 的 venv 运行")
    return pd


def _load_api_settings():
    key = os.getenv(API_KEY_ENV, "").strip()
    base = os.getenv(BASE_URL_ENV, "").strip() or DEFAULT_BASE_URL
    return base, key


def _call_llm_json(prompt, model, base_url, api_key, temperature=0.0, retries=3):
    """调用 Chat Completions 并解析 JSON 输出，带重试。"""
    if not api_key:
        raise RuntimeError(f"缺少环境变量 {API_KEY_ENV}")
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=300)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return _extract_json(content)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(min(2 ** attempt, 10))
    raise RuntimeError(f"LLM 调用失败（{retries} 次）：{last_err}")


def _extract_json(content):
    """宽松 JSON 提取：直接解析 / 代码块 / 首对花括号。"""
    text = content.strip()
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        return json.loads(m.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError(f"无法解析 LLM 输出为 JSON：{text[:200]}")


def _column_profile(df, col, n_samples=3):
    series = df[col]
    nonnull = series.notna().sum()
    ratio = round(nonnull / len(df), 3) if len(df) else 0
    samples = [str(v)[:120] for v in series.dropna().astype(str).head(n_samples)]
    return {
        "name": str(col),
        "dtype": str(series.dtype),
        "nonnull_ratio": ratio,
        "samples": samples,
    }


# ---------------------------------------------------------------- 阶段 1 认列

def detect_schema(df, model, base_url, api_key):
    """LLM 认列，返回 manifest dict。"""
    profiles = [_column_profile(df, c) for c in df.columns]
    prompt = SCHEMA_PROMPT.format(columns=json.dumps(profiles, ensure_ascii=False, indent=2))
    manifest = _call_llm_json(prompt, model, base_url, api_key)
    for role in ("query", "answer", "search_result", "score"):
        if role not in manifest or not manifest[role]:
            raise ValueError(f"认列结果缺少必需角色 {role}：{manifest}")
    return manifest
