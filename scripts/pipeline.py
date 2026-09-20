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


# ---------------------------------------------------------------- 阶段 2 判定

def _detect_language(df, manifest):
    """根据 query 列样本粗判评测语言（zh/ru/fr/ar/id/de/en/其他）。"""
    qcol = manifest["query"]
    sample = " ".join(df[qcol].dropna().astype(str).head(100))
    if re.search(r"[\u4e00-\u9fff]", sample):
        return "zh"
    counts = {
        "ru": len(re.findall(r"[а-яА-ЯёЁ]", sample)),
        "fr": len(re.findall(r"[àâçéèêëîïôùûüÿœÀÂÇÉÈÊËÎÏÔÙÛÜŒ]", sample)),
        "ar": len(re.findall(r"[\u0600-\u06FF]", sample)),
        "id": len(re.findall(r"\b(dan|yang|untuk|dengan|adalah|apa|itu|ini)\b", sample, re.I)),
        "de": len(re.findall(r"\b(der|die|das|und|ist|nicht|mit|von|den|dem)\b", sample, re.I)),
    }
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else "en"


LANGUAGE_HINT = {
    "zh": "Chinese (中文)",
    "ru": "Russian (русский)",
    "fr": "French (français)",
    "ar": "Arabic (العربية)",
    "id": "Indonesian (Bahasa Indonesia)",
    "de": "German (Deutsch)",
    "en": "English",
}


def classify_one(rid, query_text, answer_text, nfile, language_hint, manifest, model, base_url, api_key):
    """判定单个 request_id，返回 dict。"""
    nfile_text = f"召回文件数 = {nfile}" if nfile is not None else "召回文件数未知"
    prompt = CLASSIFY_PROMPT.format(
        query=query_text[:2000] or "(空)",
        answer=answer_text[:2000] or "(空)",
        nfile_text=nfile_text,
        language_hint=language_hint,
    )
    data = _call_llm_json(prompt, model, base_url, api_key)
    return _verdict_dict(rid, data)


def _verdict_dict(rid, data):
    issues = data.get("quality_issues") or []
    if isinstance(issues, str):
        issues = [issues]
    return {
        "request_id": str(rid),
        "is_specified": data.get("is_specified", "存疑"),
        "quality_issues": issues,
        "reason": data.get("reason", ""),
        "confidence": float(data.get("confidence", 0.0) or 0.0),
        "status": "ok",
    }


def classify_batch(batch_items, language_hint, model, base_url, api_key):
    """一次 LLM 调用判定一批 request_id，返回 [(rid, verdict_dict), ...]。

    batch_items: list of (rid_s, qtext, atext, nfile)
    """
    qs = []
    for i, (rid, q, a, nf) in enumerate(batch_items):
        nfile_text = f"召回文件数 = {nf}" if nf is not None else "召回文件数未知"
        qs.append({
            "index": i,
            "query": (q[:2000] or "(空)"),
            "answer": (a[:2000] or "(空)"),
            "nfile": nfile_text,
        })
    prompt = CLASSIFY_BATCH_PROMPT.format(
        questions=json.dumps(qs, ensure_ascii=False),
        language_hint=language_hint,
    )
    data = _call_llm_json(prompt, model, base_url, api_key)
    results = data.get("results") or []
    if len(results) != len(batch_items):
        raise ValueError(f"批量判定条数不符: 返回 {len(results)} != 输入 {len(batch_items)}")
    index_of = {}
    for i, item in enumerate(batch_items):
        index_of[i] = item
    out = []
    for res in results:
        idx = res.get("index")
        if idx not in index_of:
            continue
        rid, q, a, nf = index_of[idx]
        out.append((rid, _verdict_dict(rid, res)))
    if len(out) != len(batch_items):
        raise ValueError(f"批量判定 index 匹配不全: {len(out)} != {len(batch_items)}")
    return out


def _load_cache(cache_path):
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cache[str(rec.get("request_id"))] = rec
    return cache


def _append_cache(cache_path, rec):
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def classify_all(df, manifest, model, base_url, api_key, cache_path, workers=10, batch_size=20):
    """按 request_id 分组判定（批量调用），缓存续跑。返回 {request_id: rec}。"""
    qcol, acol = manifest["query"], manifest["answer"]
    cache = _load_cache(cache_path)
    fncol = manifest.get("filename")
    items = []
    for rid, grp in df.groupby(manifest["request_id"] if manifest.get("request_id") else qcol, sort=False):
        rid_s = str(rid)
        if rid_s in cache and cache[rid_s].get("status") == "ok":
            continue
        qtext = str(grp[qcol].dropna().iloc[0]) if grp[qcol].notna().any() else ""
        atext = str(grp[acol].dropna().iloc[0]) if acol in grp and grp[acol].notna().any() else ""
        nfile = None
        if fncol and fncol in grp:
            files = {str(v) for v in grp[fncol].dropna() if str(v).strip()}
            nfile = len(files)
        items.append((rid_s, qtext, atext, nfile))
    lang = _detect_language(df, manifest)
    language_hint = LANGUAGE_HINT.get(lang, lang)
    print(f"  评测语言判定: {language_hint}")
    if items:
        n_batch = (len(items) + batch_size - 1) // batch_size
        print(f"  待判定 {len(items)} 条 → {n_batch} 批（batch={batch_size}，并发批数≤{workers}）缓存命中 {len(cache)}")
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(classify_batch, b, language_hint, model, base_url, api_key): b
                for b in batches
            }
            for fut in as_completed(futures):
                b = futures[fut]
                try:
                    results = fut.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"    ⚠ 批次失败（{len(b)} 条）：{exc}")
                    for rid, *_ in b:
                        rec = {"request_id": rid, "status": "error", "error": str(exc)}
                        cache[rid] = rec
                        _append_cache(cache_path, rec)
                    continue
                for rid, rec in results:
                    cache[rid] = rec
                    _append_cache(cache_path, rec)
                print(f"    {len(results)} 条完成（本批 {[r[:24] + '...' for r, *_ in b]}）")
    else:
        print(f"  全部命中缓存，跳过判定（{len(cache)} 条）")
    return cache


# ---------------------------------------------------------------- 阶段 3 剔除

def apply_filter(df, manifest, verdicts, gray_verdicts=None):
    """确定性剔除：指定/质量问题/灰色裁决/chunk 全空。返回 (candidates, removed)。"""
    qcol = manifest["query"]
    scur = manifest["search_result"]
    gray_verdicts = gray_verdicts or {}
    removed = {}  # request_id -> [原因]
    keeps = set()
    for rid, grp in df.groupby(manifest["request_id"] if manifest.get("request_id") else qcol, sort=False):
        rid_s = str(rid)
        rec = verdicts.get(rid_s, {})
        reasons = []
        if gray_verdicts.get(rid_s) == "指定":
            reasons.append("灰色人工裁决:指定")
        if rec.get("status") != "ok":
            reasons.append(f"判定失败: {rec.get('error', 'unknown')}")
        spec = rec.get("is_specified")
        if spec == "是":
            reasons.append("指定query")
        elif spec == "存疑":
            gv = gray_verdicts.get(rid_s)
            if gv == "指定":
                reasons.append("灰色人工裁决:指定")
            elif gv == "保留":
                pass
            else:
                reasons.append("灰色待裁决")
        for issue in rec.get("quality_issues") or []:
            reasons.append(f"质量:{issue}")
        # chunk 全空兜底
        if scur in grp and grp[scur].apply(lambda v: pd.isna(v) or (isinstance(v, str) and not v.strip())).all():
            reasons.append("chunk全空")
        if reasons:
            removed[rid_s] = reasons
        else:
            keeps.add(rid_s)
    cand = df[df[manifest["request_id"] if manifest.get("request_id") else qcol].astype(str).isin(keeps)]
    return cand, removed


def summarize_filter_stats(removed: dict, n_input: int) -> dict:
    """Aggregate removal reasons into rates for eval readers (P2.6).

    Fail-closed removals stay fail-closed; this only makes the funnel visible.
    Also records ``difficulty_shift`` noting that prefer-存疑 (when unsure)
    enriches the gray/uncertain pool rather than silently forcing 是.
    """
    from collections import Counter
    reason_counts: Counter = Counter()
    for reasons in removed.values():
        for r in reasons:
            # normalize prefixes
            if r.startswith("判定失败"):
                key = "classify_failure"
            elif r.startswith("灰色人工裁决"):
                key = "gray_human_specified"
            elif r == "指定query":
                key = "specified_query"
            elif r == "灰色待裁决":
                key = "gray_pending"
            elif r.startswith("质量:"):
                key = f"quality:{r.split(':', 1)[1]}"
            elif r == "chunk全空":
                key = "chunk_empty"
            else:
                key = r
            reason_counts[key] += 1
    n_removed = len(removed)
    n_kept = max(0, n_input - n_removed)
    rates = {k: round(v / n_input, 4) if n_input else 0.0 for k, v in reason_counts.items()}
    return {
        "n_input": n_input,
        "n_removed": n_removed,
        "n_kept": n_kept,
        "removal_rate": round(n_removed / n_input, 4) if n_input else 0.0,
        "reason_counts": dict(reason_counts),
        "reason_rates": rates,
        "difficulty_shift": {
            "tie_break_policy": "prefer_uncertain",
            "note": (
                "When the classifier is unsure between 是/否, it must emit 存疑 "
                "(uncertain) rather than forcing 是. Uncertain rows stay fail-closed "
                "(removed as 灰色待裁决 until human verdict). This shifts difficulty "
                "visibility into the gray pool instead of silently enriching the "
                "removed-as-specified set."
            ),
        },
    }



# ---------------------------------------------------------------- 阶段 4 抽样

def stratified_sample(index_ids, rows_by_id, strata, seed, n):
    """按召回行数分层配额随机抽样，返回选中 request_id 列表。"""
    groups = {}
    for lo, hi in strata:
        groups[(lo, hi)] = [rid for rid in index_ids if lo <= rows_by_id[rid] <= hi]
    total = len(index_ids)
    quota = {}
    for g, ids in groups.items():
        quota[g] = int(n * len(ids) / total)
    leftover = n - sum(quota.values())
    if leftover > 0:
        order = sorted(groups, key=lambda g: (len(groups[g]) - quota[g]), reverse=True)
        for g in order:
            if leftover == 0:
                break
            quota[g] += 1
            leftover -= 1
    rng = random.Random(seed)
    selected = []
    for g, ids in groups.items():
        selected.extend(rng.sample(ids, quota[g]))
    return selected


# ---------------------------------------------------------------- 阶段 5 转 zhfmt

def to_zhfmt(df, manifest):
    """长格式 → 6 列块状（块首行填 query/答案，块内 score 降序）。"""
    qcol = manifest["query"]
    acol = manifest["answer"]
    scur = manifest["search_result"]
    score_col = manifest["score"]
    fncol = manifest.get("filename")
    refcol = manifest.get("reference")
    rows = []
    for rid, blk in df.groupby(manifest["request_id"] if manifest.get("request_id") else qcol, sort=False):
        blk = blk.sort_values(score_col, ascending=False)
        first = True
        for _, r in blk.iterrows():
            rows.append({
                "query": r[qcol] if first else None,
                "标注": (r[refcol] if refcol in blk and pd.notna(r[refcol]) else None),
                "filename": (r[fncol] if fncol in blk and pd.notna(r[fncol]) else None),
                "搜索结果": (r[scur] if pd.notna(r[scur]) else None),
                "score": r[score_col],
                "大模型问答结果": (r[acol] if first and pd.notna(r[acol]) else None),
            })
            first = False
    return pd.DataFrame(rows, columns=ZH_COLS)


# ---------------------------------------------------------------- 主流程

def main():
    parser = argparse.ArgumentParser(description="KB query pipeline: 认列→判定→剔除→抽样→转zhfmt")
    parser.add_argument("--file", required=True, help="输入文件（xlsx/csv）")
    parser.add_argument("--out-dir", default=None, help="输出目录（默认=输入文件同目录）")
    parser.add_argument("--model", default=os.getenv("KB_QA_MODEL", "alibaba/qwen3.7-plus"), help="LLM 模型")
    parser.add_argument("--workers", type=int, default=10, help="判定并发批数")
    parser.add_argument("--n", type=int, default=100, help="抽样条数")
    parser.add_argument("--batch-size", type=int, default=20, help="批量判定：一次 LLM 调用判多少条")
    parser.add_argument("--seed", type=int, default=42, help="抽样随机种子")
    parser.add_argument("--schema-yes", action="store_true", help="跳过认列人工确认")
    parser.add_argument("--schema-override", default=None, help="人工修正映射，如 query=问题;answer=回答")
    parser.add_argument("--gray-verdicts", default=None, help="灰色裁决 csv: request_id,裁决(指定|保留)")
    parser.add_argument("--skip-stages", default="", help="逗号分隔跳过的阶段编号(1-5)")
    parser.add_argument("--resume", action="store_true", default=True, help="断点续跑（默认开启）")
    args = parser.parse_args()

    pd = _require_pd()
    base_url, api_key = _load_api_settings()
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.file))
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.file))[0]
    skip = {s.strip() for s in args.skip_stages.split(",") if s.strip()}

    # 读文件
    if args.file.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(args.file, sheet_name="Results") if "Results" in pd.ExcelFile(args.file).sheet_names else pd.read_excel(args.file)
    elif args.file.lower().endswith(".csv"):
        df = pd.read_csv(args.file)
    else:
        raise ValueError(f"不支持的文件格式: {args.file}")
    print(f"输入: {args.file}  行数={len(df)}  列={list(df.columns)}")

    manifest = None
    manifest_path = os.path.join(out_dir, "01_schema_manifest.json")

    # 阶段 1：认列
    if "1" not in skip:
        if args.schema_override:
            manifest = dict(kv.split("=", 1) for kv in args.schema_override.split(";"))
            print("使用人工映射:", manifest)
        elif os.path.exists(manifest_path) and args.resume:
            manifest = json.load(open(manifest_path, encoding="utf-8"))
            print("复用已有 manifest:", manifest)
        else:
            print("[阶段1] LLM 认列...")
            manifest = detect_schema(df, args.model, base_url, api_key)
            print("认列结果:", json.dumps(manifest, ensure_ascii=False, indent=2))
            if not args.schema_yes:
                ans = input("映射正确吗？[Y/n] ").strip().lower()
                if ans in ("n", "no"):
                    print("使用 --schema-override 提供修正映射后重跑。")
                    sys.exit(2)
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
    else:
        manifest = json.load(open(manifest_path, encoding="utf-8"))
    print(f"列映射: {json.dumps({k: v for k, v in manifest.items() if k in ('query','answer','search_result','score','filename','request_id')}, ensure_ascii=False)}")

    # 阶段 2：判定（缓存续跑）
    classified = {}
    if "2" not in skip:
        print("[阶段2] LLM 逐 request_id 判定...")
        cache_path = os.path.join(out_dir, f"{base}.classify_cache.jsonl")
        classified = classify_all(df, manifest, args.model, base_url, api_key, cache_path, workers=args.workers, batch_size=args.batch_size)
    else:
        cache_path = os.path.join(out_dir, f"{base}.classify_cache.jsonl")
        classified = _load_cache(cache_path)

    # 灰色清单
    gray = {rid: rec for rid, rec in classified.items() if rec.get("is_specified") == "存疑"}
    gray_path = os.path.join(out_dir, f"{base}.gray_list.csv")
    if gray:
        key = manifest["request_id"] if manifest.get("request_id") else manifest["query"]
        qmap = df.groupby(key)[manifest["query"]].first().astype(str).to_dict()
        rows = [{"request_id": rid, "query": qmap.get(rid, ""), "reason": rec.get("reason", "")} for rid, rec in gray.items()]
        pd.DataFrame(rows).to_csv(gray_path, index=False, encoding="utf-8-sig")
        print(f"⚠ 存疑 {len(gray)} 条 → {gray_path}；用 --gray-verdicts 提供裁决（列: request_id,裁决=指定|保留）")
    gray_verdicts = {}
    if args.gray_verdicts and os.path.exists(args.gray_verdicts):
        gv = pd.read_csv(args.gray_verdicts, encoding="utf-8-sig")
        gray_verdicts = {str(r["request_id"]): str(r["裁决"]) for _, r in gv.iterrows()}

    # 阶段 3：剔除
    if "3" not in skip:
        print("[阶段3] 确定性剔除...")
        cand, removed = apply_filter(df, manifest, classified, gray_verdicts)
        cand_path = os.path.join(out_dir, f"{base}_candidates.xlsx")
        cand.to_excel(cand_path, index=False, sheet_name="Results")
        removed_df = pd.DataFrame([{"request_id": rid, "原因": ";".join(rs)} for rid, rs in removed.items()])
        removed_df.to_csv(os.path.join(out_dir, f"{base}.removed_list.csv"), index=False, encoding="utf-8-sig")
        # P2.6: always emit filter_stats.json with removal rates + difficulty_shift note
        key_col = manifest["request_id"] if manifest.get("request_id") else manifest["query"]
        n_input = int(df[key_col].astype(str).nunique())
        filter_stats = summarize_filter_stats(removed, n_input)
        stats_path = os.path.join(out_dir, f"{base}.filter_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(filter_stats, f, ensure_ascii=False, indent=2)
        print(f"  候选 {len(cand)} 行 / {cand[manifest['request_id']].nunique() if manifest.get('request_id') else '?'} 唯一  剔除 {len(removed)} 条 → {cand_path}")
        print(f"  filter_stats → {stats_path}  removal_rate={filter_stats['removal_rate']:.1%}  reasons={filter_stats['reason_counts']}")
        print(f"  difficulty_shift: {filter_stats['difficulty_shift']['tie_break_policy']} — {filter_stats['difficulty_shift']['note'][:80]}…")
    else:
        cand_path = os.path.join(out_dir, f"{base}_candidates.xlsx")
        if not os.path.exists(cand_path):
            raise FileNotFoundError(f"跳过阶段3但产物不存在: {cand_path}（需先运行阶段3）")
        cand = pd.read_excel(cand_path, sheet_name="Results")

    # 阶段 4：抽样
    if "4" not in skip:
        print(f"[阶段4] 分层抽样 n={args.n} seed={args.seed}...")
        key = manifest["request_id"] if manifest.get("request_id") else manifest["query"]
        rows_by_id = df.groupby(key).size().to_dict()
        cand_ids = cand[key].astype(str).unique().tolist()
        strata = [(1, 2), (3, 5), (6, 10 ** 9)]
        sel = stratified_sample(cand_ids, rows_by_id, strata, args.seed, args.n)
        if len(sel) != args.n:
            print(f"⚠ 抽样数 {len(sel)} != {args.n}（候选不足？）")
        sampled = cand[cand[key].astype(str).isin(sel)]
        samp_path = os.path.join(out_dir, f"{base}_sampled.xlsx")
        sampled.to_excel(samp_path, index=False, sheet_name="Results")
        print(f"  抽样 {len(sel)} 条 → {samp_path}  唯一性={sampled[key].nunique()}")
    else:
        samp_path = os.path.join(out_dir, f"{base}_sampled.xlsx")
        if not os.path.exists(samp_path):
            raise FileNotFoundError(f"跳过阶段4但产物不存在: {samp_path}（需先运行阶段4）")
        sampled = pd.read_excel(samp_path, sheet_name="Results")

    # 阶段 5：转 zhfmt
    if "5" not in skip:
        print("[阶段5] 转 zhfmt 6列块状...")
        zh = to_zhfmt(sampled, manifest)
        zh_path = os.path.join(out_dir, f"{base}_zhfmt.xlsx")
        zh.to_excel(zh_path, index=False, sheet_name="ZH_Sample")
        print(f"  → {zh_path}  总行={len(zh)}  块数={zh['query'].notna().sum()}")

    print("完成。")


if __name__ == "__main__":
    main()
