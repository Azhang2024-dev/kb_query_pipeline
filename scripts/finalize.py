# -*- coding: utf-8 -*-
"""kb-query-pipeline 后处理：抽样后的人工复核闭环。

子命令：
  dedup       语义去重报告：Jaccard 粗筛重复对（可选 --llm-check 复核边界对）
  refill      差额补抽：从候选池补抽 N 条，排除现有池/待删/相似/指定模式
  build       生成最终产物：按 id 清单从候选池抽行 → final100.xlsx + zhfmt_final.xlsx
  final-check 最终校验：块数=100 / 无重复 request_id / 无指定残留 / 无重复对

设计原则：确定性逻辑在此，重试/多轮交互/人工确认由 agent 编排层负责。
"""
import argparse
import json
import os
import random
import re
import sys
import unicodedata

sys.stdout.reconfigure(encoding="utf-8")

try:
    import pandas as pd
except ImportError:
    pd = None

from pipeline import stratified_sample, to_zhfmt, _call_llm_json, _load_api_settings  # noqa: E402

DEFAULT_TH = 0.3
DEFAULT_SEED = 42
DEFAULT_STRATA = [(1, 2), (3, 5), (6, 10 ** 9)]


# ---------------------------------------------------------------- 文本工具

def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).replace("\u2019", "'").replace("\u2018", "'")
    return re.sub(r"\s+", " ", s).strip().lower()


def tokens(s):
    return set(re.findall(r"[a-zа-яё0-9\u0600-\u06ff]+", norm(s)))


def jaccard(a, b):
    u = a | b
    return len(a & b) / len(u) if u else 0


def load_df(path):
    if str(path).lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path, sheet_name="Results") if "Results" in pd.ExcelFile(path).sheet_names else pd.read_excel(path)
    return pd.read_csv(path)


def _manifest(out_dir, base):
    p = os.path.join(out_dir, "01_schema_manifest.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"manifest 不存在: {p}")
    return json.load(open(p, encoding="utf-8"))


def _key(manifest):
    return manifest["request_id"] if manifest.get("request_id") else manifest["query"]


# ---------------------------------------------------------------- dedup

def dedup_report(pool_df, manifest, th=DEFAULT_TH, top_n=50):
    """Jaccard 粗筛重复对，返回 [(sim, q1, q2, rid1, rid2), ...]。"""
    qcol = manifest["query"]
    key = _key(manifest)
    uniq = pool_df.drop_duplicates(key)
    qs = list(uniq[qcol].astype(str))
    rids = list(uniq[key].astype(str))
    toks = [tokens(q) for q in qs]
    pairs = []
    for i in range(len(toks)):
        for j in range(i + 1, len(toks)):
            s = jaccard(toks[i], toks[j])
            if s >= th:
                pairs.append((round(s, 3), qs[i], qs[j], rids[i], rids[j]))
    pairs.sort(reverse=True)
    return pairs[:top_n]


def llm_check_pairs(pairs, model, base_url, api_key, workers=4):
    """LLM 复核边界对：把 (sim, q1, q2) 判为真重复/误报。返回带 is_dup 的列表。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from prompts import DEDUP_PROMPT

    def check(p):
        prompt = DEDUP_PROMPT.format(q1=p[1], q2=p[2])
        data = _call_llm_json(prompt, model, base_url, api_key)
        return p + (bool(data.get("is_duplicate")), data.get("reason", ""))

    out = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(check, p) for p in pairs]
        for fut in as_completed(futures):
            out.append(fut.result())
    out.sort(key=lambda x: x[0], reverse=True)
    return out


# ---------------------------------------------------------------- refill（差额补抽）

BAD_PATTERNS_DEFAULT = {
    "fr": [r"contacter\s+[A-ZÀ-Ý][a-zà-ÿ'’ -]*\s*par\s+(courriel|e-?mail)", r"\bCNSS\b",
           r"stages?\s+de\s+recherche", r"\bHAZOP\b", r"\bTraccar\b", r"forclusion",
           r"baccalauréat", r"vendue\s*à\s*1000", r"\bDEMBA\b", r"H335"],
    "ar": [r"الدور\s+الثاني", r"التظلمات", r"الأزهرية", r"كفاية\s+الأصول", r"الثالث\s+المتوسط"],
    "ru": [r"код\s+31", r"больничн", r"котельной"],
}


def refill(cand_df, manifest, keep_ids, n=100, seed=DEFAULT_SEED, th=DEFAULT_TH,
           bad_patterns=None, exclude_ids=None):
    """从候选池补抽 n 条，返回选中 request_id 列表。

    排除：keep_ids（已保留） + exclude_ids（待删/已否决）
    过滤：与保留池语义相似 < th；与保留池 + 已选彼此去重；命中指定模式黑名单则排除。
    """
    qcol = manifest["query"]
    key = _key(manifest)
    bad_patterns = bad_patterns or {}
    exclude_ids = set(exclude_ids or [])
    excl = set(keep_ids) | exclude_ids

    pool = cand_df[~cand_df[key].astype(str).isin(excl)].drop_duplicates(key)
    ref_df = cand_df[cand_df[key].astype(str).isin(keep_ids)].drop_duplicates(key)
    ref_toks = [tokens(q) for q in ref_df[qcol]]

    def best_sim(t):
        return max((jaccard(t, kt) for kt in ref_toks), default=0)

    pool = pool.copy()
    pool["_sim"] = pool[qcol].map(lambda q: best_sim(tokens(q)))
    qual = pool[pool["_sim"] < th].sort_values("_sim", ascending=False)

    bad = []
    if isinstance(bad_patterns, dict):
        flat = []
        for pats in bad_patterns.values():
            flat.extend(pats)
        bad = [re.compile(p, re.IGNORECASE) for p in flat]
    if bad:
        qual = qual[~qual[qcol].apply(lambda q: any(b.search(str(q)) for b in bad))]

    used_toks = list(ref_toks)
    sel = []
    for _, r in qual.iterrows():
        t = tokens(r[qcol])
        if all(jaccard(t, u) < th for u in used_toks):
            sel.append(r)
            used_toks.append(t)

    sdf = pd.DataFrame(sel)
    if len(sdf) < n:
        print(f"  ⚠ 合格候选 {len(sdf)} < 需补 {n}（候选池可能不足）")
    rows_by_id = cand_df.groupby(key).size()
    picked = stratified_sample(list(sdf[key].astype(str)), rows_by_id, DEFAULT_STRATA, seed, min(n, len(sdf)))
    return sdf[sdf[key].astype(str).isin(picked)]


# ---------------------------------------------------------------- build

def build_final(cand_df, manifest, ids, out_path_xlsx, out_path_zhfmt):
    key = _key(manifest)
    final = cand_df[cand_df[key].astype(str).isin(set(ids))]
    final.to_excel(out_path_xlsx, index=False)
    zh = to_zhfmt(final, manifest)
    zh.to_excel(out_path_zhfmt, index=False, sheet_name="ZH_Sample")
    return final, zh


# ---------------------------------------------------------------- final-check

def final_check(final_df, manifest, zhfmt_df=None, bad_patterns=None, th=DEFAULT_TH):
    key = _key(manifest)
    qcol = manifest["query"]
    uniq = final_df.drop_duplicates(key)
    n_uniq = uniq[key].nunique()
    n_rows = len(final_df)
    problems = []
    if n_uniq != 100:
        problems.append(f"唯一 request_id = {n_uniq} != 100")
    # 语义重复对
    qs = list(uniq[qcol].astype(str))
    toks = [tokens(q) for q in qs]
    dups = []
    for i in range(len(toks)):
        for j in range(i + 1, len(toks)):
            s = jaccard(toks[i], toks[j])
            if s >= th:
                dups.append((round(s, 3), qs[i], qs[j]))
    if dups:
        problems.append(f"重复对 {len(dups)} 对：{dups[:5]}…")
    # 指定模式残留（粗检）
    if bad_patterns:
        flat = []
        for pats in bad_patterns.values():
            flat.extend(pats)
        bad = [re.compile(p, re.IGNORECASE) for p in flat]
        hits = [q for q in qs if any(b.search(q) for b in bad)]
        if hits:
            problems.append(f"疑似指定模式残留 {len(hits)} 条：{hits[:5]}…")
    return {"n_unique": n_uniq, "n_rows": n_rows, "dups": dups, "problems": problems}


# ---------------------------------------------------------------- CLI

def main():
    parser = argparse.ArgumentParser(description="KB query pipeline 后处理：复核/补抽/校验")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("dedup", help="语义去重报告")
    p.add_argument("--pool", required=True, help="池文件（xlsx/csv，已抽样或最终）")
    p.add_argument("--out-dir", required=True, help="输出目录（含 01_schema_manifest.json）")
    p.add_argument("--th", type=float, default=DEFAULT_TH)
    p.add_argument("--llm-check", action="store_true", help="用 LLM 复核边界对")
    p.add_argument("--workers", type=int, default=4)

    p = sub.add_parser("refill", help="差额补抽")
    p.add_argument("--candidates", required=True, help="候选池文件")
    p.add_argument("--out-dir", required=True, help="输出目录（含 01_schema_manifest.json）")
    p.add_argument("--keep", required=True, help="保留 id 清单文件（txt 每行一个或 json 数组）")
    p.add_argument("--exclude", default=None, help="额外排除 id 清单文件（待删/已否决）")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--th", type=float, default=DEFAULT_TH)
    p.add_argument("--out", default=None, help="补抽结果输出 csv（默认打印）")

    p = sub.add_parser("build", help="按 id 清单生成 final100 + zhfmt")
    p.add_argument("--candidates", required=True, help="候选池文件")
    p.add_argument("--out-dir", required=True, help="输出目录（含 01_schema_manifest.json）")
    p.add_argument("--ids", required=True, help="最终 id 清单文件（txt 每行或 json 数组）")
    p.add_argument("--base", required=True, help="输出文件名基名（如 llm_with_evidence_fr）")

    p = sub.add_parser("final-check", help="最终校验")
    p.add_argument("--final", required=True, help="final100 文件")
    p.add_argument("--out-dir", required=True, help="输出目录（含 01_schema_manifest.json）")
    p.add_argument("--th", type=float, default=DEFAULT_TH)

    args = parser.parse_args()
    manifest = _manifest(args.out_dir, None)
    key = _key(manifest)

    if args.cmd == "dedup":
        pool = load_df(args.pool)
        pairs = dedup_report(pool, manifest, th=args.th)
        print(f"重复对（sim≥{args.th}）：{len(pairs)}")
        if args.llm_check and pairs:
            base_url, api_key = _load_api_settings()
            model = os.getenv("KB_QA_MODEL", "azure/gpt-5.4")
            pairs = llm_check_pairs(pairs, model, base_url, api_key, workers=args.workers)
            for s, q1, q2, r1, r2, dup, reason in pairs:
                print(f"  [sim={s}] {'重复' if dup else '误报'} {r1}|{r2}: {q1[:40]} || {q2[:40]}  ({reason})")
        else:
            for s, q1, q2, r1, r2 in pairs:
                print(f"  [sim={s}] {r1}|{r2}: {q1[:40]} || {q2[:40]}")

    elif args.cmd == "refill":
        cand = load_df(args.candidates)
        keep_ids = [l.strip() for l in open(args.keep, encoding="utf-8") if l.strip()]
        exclude_ids = []
        if args.exclude:
            exclude_ids = [l.strip() for l in open(args.exclude, encoding="utf-8") if l.strip()]
        picked = refill(cand, manifest, keep_ids, n=args.n, seed=args.seed, th=args.th,
                        bad_patterns=BAD_PATTERNS_DEFAULT, exclude_ids=exclude_ids)
        qcol = manifest["query"]
        print(f"补抽 {len(picked)} 条：")
        for _, r in picked.iterrows():
            print(f"  {r[key]}  |  {r[qcol]}")
        if args.out:
            picked.to_csv(args.out, index=False, encoding="utf-8-sig")
            print(f"→ {args.out}")

    elif args.cmd == "build":
        cand = load_df(args.candidates)
        ids = [l.strip() for l in open(args.ids, encoding="utf-8") if l.strip()]
        f100 = os.path.join(args.out_dir, f"{args.base}_final100.xlsx")
        zh = os.path.join(args.out_dir, f"{args.base}_zhfmt_final.xlsx")
        final, zdf = build_final(cand, manifest, ids, f100, zh)
        print(f"final100: {len(final)} 行 / {final[key].nunique()} 唯一 → {f100}")
        print(f"zhfmt: {len(zdf)} 行 / {zdf['query'].notna().sum()} 块 → {zh}")

    elif args.cmd == "final-check":
        final = load_df(args.final)
        rep = final_check(final, manifest, th=args.th, bad_patterns=BAD_PATTERNS_DEFAULT)
        print(f"唯一 id = {rep['n_unique']} / 行 = {rep['n_rows']}")
        if rep["problems"]:
            print("⚠ 问题：")
            for p_ in rep["problems"]:
                print(f"  - {p_}")
        else:
            print("✅ 校验通过")


if __name__ == "__main__":
    main()
