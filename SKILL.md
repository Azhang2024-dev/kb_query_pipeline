---
name: kb-query-pipeline
description: "Knowledge base query filtering & sampling pipeline (知识库 query 筛选抽样流水线): LLM 自动识别任意表头列映射 (schema detection), 逐批判定是否指定 query (指示词/具名实体指向特定对象) 与质量问题, 确定性剔除 + 分层随机抽样 + 转 6 列块状 zhfmt 格式, 抽样后人工复核闭环 (去重报告/差额补抽/最终校验)。换表头零改码, 断点续跑, 抽样 seed 可复现。当用户要给 KB 评测数据做筛选/抽样/格式转换, 处理换表头的检索结果文件, 担心漏掉指定 query, 或想剔除空回答/语言混杂/无召回数据/语义重复时使用。"
---

# KB Query Pipeline (kb-query-pipeline)

## Overview

知识库问答评测数据的预处理流水线，两段式：

```
┌─ 自动链 pipeline.py ─────────────────────────────┐
│ ① 认列(LLM) → ② 判定(LLM批量+缓存) → ③ 剔除(确定性)  │
│ → ④ 分层抽样(确定性) → ⑤ 转zhfmt(确定性)             │
└──────────────────┬───────────────────────────────┘
                   │ 抽样结果交人工过目（标黄/标记）
┌──────────────────▼───────────────────────────────┐
│ finalize.py 人工复核闭环：                          │
│ dedup 去重报告 → 人工确认删哪条 → refill 差额补抽      │
│ → build 生成最终产物 → final-check 最终校验           │
└──────────────────────────────────────────────────┘
```

- **LLM 负责语义**：认列（任意表头/任意语言列名→语义映射）、逐批判定"是否指定 query + 质量问题"并附理由。
- **确定性代码负责机械**：剔除、分层抽样、格式转换、去重报告、补抽、校验——可复现、可测试。
- **人工复核兜底**：LLM 判定无法保证 100% 覆盖，抽样后必须人工过目，标黄项回填后差额补抽。
- **语言无关**：prompt 用自然语言描述规则，不含任何语言词表；变格、省音、同义表达 LLM 均可识别。

## When to Use

- 评测数据换了一批表头/语言，不想改代码重新筛选
- 需要剔除"指定 query"（指示词指向 或 具名实体指向特定对象）
- 需要剔除质量问题（空回答、语言混杂、chunk 全空、不完整问题）
- 需要按召回行数分层随机抽样 N 条（seed 固定可复现）
- 需要把长格式检索结果转成 6 列块状 zhfmt（query/标注/filename/搜索结果/score/大模型问答结果）
- 抽样后需要去重报告、差额补抽、最终校验（人工复核闭环）

**Do NOT use when:** 需要评测打分（用 kb-qa-eval）、仅做简单文件格式转换不涉筛选。

## Requirements

- Python venv：`D:\\Users\\admin\\.agents\\skills\\kb-qa-eval\\venv\\Scripts\\python.exe`（含 pandas/openpyxl/requests）
- 环境变量：
  - `KB_QA_API_KEY`（必填）
  - `KB_QA_API_BASE_URL`（可选，默认 `https://maragate.aibotplatform.com/v1`）
  - `KB_QA_MODEL`（可选，默认 `azure/gpt-5.4`；**换模型 = 删旧 `*.classify_cache.jsonl` 重新判定**）

## Usage — 自动链（pipeline.py）

```powershell
$env:KB_QA_API_KEY = "<key>"
python scripts/pipeline.py --file <输入.xlsx> [--out-dir <目录>] [--model M] [--workers 10] [--batch-size 20] [--n 100] [--seed 42] [--schema-yes]
```

### 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--file` | 必填 | 输入 xlsx/csv（自动读 `Results` sheet 或首个 sheet） |
| `--out-dir` | 输入文件目录 | 输出目录 |
| `--model` | `KB_QA_MODEL` / azure/gpt-5.4 | 判定/认列模型 |
| `--workers` | 10 | 判定并发批数 |
| `--batch-size` | 20 | 批量判定：一次 LLM 调用判多少条（性能关键） |
| `--n` | 100 | 抽样条数 |
| `--seed` | 42 | 抽样随机种子 |
| `--schema-yes` | off | 跳过认列人工确认（Agent 自动跑时必带） |
| `--schema-override` | 无 | 人工修正映射：`query=问题;answer=回答;...` |
| `--gray-verdicts` | 无 | 灰色裁决 csv（列：`request_id,裁决`，裁决=指定\\|保留）。裁决=指定时**无条件强制剔除**，优先级高于 LLM 判定 |
| `--skip-stages` | 无 | 跳过的阶段，如 `1,5` |
| `--resume` | on | 断点续跑（判定按 request_id 缓存） |

## 五阶段详解（pipeline.py）

### 阶段 1 认列（LLM，1 次调用）
读表头 + 每列 dtype/非空比例/3-5 条样本值 → LLM 返回列映射（query/answer/search_result/score/filename/request_id/...）+ 每列置信度 + 理由。
**护栏①**：映射结果打印给用户确认（Y/N）。Agent 自动跑必须带 `--schema-yes`；映射错可 `--schema-override` 人工修正。
产物：`01_schema_manifest.json`（复用）。

### 阶段 2 判定（LLM 批量并发）
按 request_id 分批（默认 20 条/批）判多维度：`is_specified`(是/否/存疑) + `quality_issues`(empty_answer/mixed_language/incomplete_question/retrieval_failed) + `reason` + `confidence`。
**断点续跑**：每批结果逐条追加写 `<base>.classify_cache.jsonl`；重跑跳过已判定 request_id，只补跑缺失/`status=error` 的。中断后重跑只花中断点之后的钱。
**护栏②**：`存疑` 条导出 `<base>.gray_list.csv`（含 query/理由），人工裁决后 `--gray-verdicts` 回填（指定→剔除，保留→放行）。
> **强制覆盖**：`--gray-verdicts` 裁决=指定对**任意判定结果**都强制剔除（含 LLM 判\"否\"的漏网指定 query），不限于`存疑`。用于人工复核发现 LLM 漏判时回填；保留裁决仅影响`存疑`条放行。

### 阶段 3 剔除（确定性）
- `is_specified=是` → 剔除（原因\"指定query\"）
- 命中 quality_issues → 剔除
- `存疑` 未裁决 → 剔除（\"灰色待裁决\"）；已裁决按裁决走
- **`--gray-verdicts` 裁决=指定 → 强制剔除**（原因\"灰色人工裁决:指定\"，覆盖 LLM 判定）
- 兜底：`search_result` 列 chunk 全空 → 剔除
产物：`<base>_candidates.xlsx` + `<base>.removed_list.csv`（每项附原因）。

### 阶段 4 分层抽样（确定性）
按召回行数分层 `1-2 / 3-5 / 6+` 配额随机抽样 N，seed 固定 → 完全可复现。**护栏④**。
产物：`<base>_sampled.xlsx` + 唯一性校验输出。

### 阶段 5 转 zhfmt（确定性）
长格式 → 6 列块状 `query/标注/filename/搜索结果/score/大模型问答结果`；块首行填 query/答案，块内 score 降序，`标注` 保留原值。答案列名取自 manifest（兼容 ar=`模型结果`/fr=`知识库问答结果`/ru=`问答结果` 等任意命名）。
产物：`<base>_zhfmt.xlsx`（sheet=`ZH_Sample`）。

## Usage — 人工复核闭环（finalize.py）

```powershell
# 1. 语义去重报告（对抽样池或最终池）
python scripts/finalize.py dedup --pool <池.xlsx> --out-dir <数据目录> [--th 0.3] [--llm-check]

# 2. 差额补抽（删除标黄/重复项后补到 N）
python scripts/finalize.py refill --candidates <candidates.xlsx> --out-dir <数据目录> `
  --keep <保留清单.txt> [--exclude <额外排除.txt>] --n 100 --seed 42 --th 0.3

# 3. 生成最终产物
python scripts/finalize.py build --candidates <candidates.xlsx> --out-dir <数据目录> `
  --ids <最终清单.txt> --base llm_with_evidence_fr

# 4. 最终校验
python scripts/finalize.py final-check --final <final100.xlsx> --out-dir <数据目录> --th 0.3
```

| 子命令 | 说明 |
|--------|------|
| `dedup` | Jaccard 粗筛重复对（`--th` 默认 0.3）；`--llm-check` 用 LLM 复核边界对（真重复/误报）。**输出供人工确认删哪条，工具不自动删** |
| `refill` | 从候选池补抽 N 条。自动排除：keep 清单 + exclude 清单 + 与保留池语义相似≥阈值 + 命中指定模式黑名单（人名联系/HAZOP/Traccar/考试年份等）。seed 固定可复现 |
| `build` | 按 id 清单从候选池抽行 → `<base>_final100.xlsx` + `<base>_zhfmt_final.xlsx` |
| `final-check` | 校验：唯一 id=100 / 无重复对(≥th) / 无指定模式残留。**疑似重复对需人工判定（含句式误报）** |

## 判定口径（严格版）—— 人工复核时的参照标准

**指定 query = 指向单一特定对象且不可泛化**，两类：

**A. 指示词指向**：this/that、ce/cet/cette/ces、этот/данный/настоящий/указанный（含变格）、هذا/هذه/المذكور、ini/itu、这/那/此/该/本 —— 有指示词即指定。

**B. 具名实体指向**（即使无指示词，问\"那个具体对象\"即指定）：
- 具体人名/邮箱：联系 Sarah Estelle Touvoli
- 具体案件/企业：forclusion 案、Cas DEMBA SARL
- 具体项目/地点：HAZOP Gamba-Ivinga et Atora
- 具体系统/平台：Traccar 服务器
- 具体考试/年份：baccalauréat général 2024、高中二轮成绩申诉
- 具体日期/区间：2024-12-02 至 2024-01-18 的实习
- 具体表格/文书/代码：硕士推荐表、病假条代码 31、H335
- 具体金额/场景：1000 法郎的薯条份量
- 具体书籍/教科书：كتاب كفاية الأصول、《三年级科学书》

**例外（非指定）**：问通用规则/知识，即使提到具名机构——\"中国驻多哥大使馆奖学金面向哪些学生\"（通用规则）≠ \"联系具体某人的邮箱\"；\"科特迪瓦政体\"、\"循环经济概念\"、\"中性线截面\"（通用知识）；今年/今天；X 是什么（定义）；это X или нет（分类）。

**仲裁规则**：不读任何具体文档、仅凭常识就能回答 → 非指定；必须看某个具体对象才能知道指谁 → 指定。**两可时判\"是\"（宁多删，可恢复；漏删会污染最终集）**。

## 黄金回归用例（护栏③）

判定 prompt 内置断言覆盖的边界（tests 自动检查 prompt 含这些内容，防 prompt 改坏）：

**必须判\"指定\"**：
- 俄语变格指示词：этой теме / этому взысканию / этим жилым помещением / этого сборника / в этом наборе / у того
- 具名实体指向（严格口径）：Sarah Estelle Touvoli 邮箱 / HAZOP Gamba-Ivinga et Atora / serveur Traccar / baccalauréat général 2024 / код 31 в больничном листе / نموذج توصية طالب الماستر / frite vendue à 1000 / كتاب كفاية الأصول / stage du 02 décembre au 18 janvier 2024

**必须判\"否\"（例外）**：
- 通用规则/知识（具名机构除外）：Ambassade de Chine au Togo 奖学金 / Côte d'Ivoire 政体 / 循环经济 / 中性线截面 / 机器翻译可发表质量 / 握手摘手套
- 时间指示：tahun ini / hari ini / 今年 / 今天
- 定义疑问：X itu apa / Qu'est-ce que X
- 分类疑问：это банковский счет или нет
- 法律术语：персональных данных / ТО 缩写

## 测试

```powershell
python tests/run_tests.py
```

覆盖：filter 判定剔除/质量/灰色裁决/chunk全空、sample seed 可复现+分层配额、convert 块状格式、cache 续跑跳过/失败重试、prompt 严格口径黄金用例（指示词+具名实体+例外）、finalize 去重/补抽/校验确定性。

## 已知边界与处理

- LLM 判定不可复现 → 灰色清单人工兜底 + `confidence` 低值提示；抽样/剔除/转换保持确定性。
- LLM 只看 query/answer/召回文件数，**看不到文件名与召回片段**，易漏判\"具名文书/具体图纸/具体病历/具体考试中心\"类指定 query → **抽样后必须人工过目**，漏网项用 `--gray-verdicts`（指定=强制剔除）或 finalize refill 回填。
- 语义重复对（同义反复，如金字塔最大/最大的金字塔）LLM 单条判不出 → finalize dedup 报告 + 人工确认删哪条。
- Jaccard 去重会误报共享句式对（\"Что такое X\" vs \"Что такое Y\"）→ 报告输出后人工判定，或用 `--llm-check` 复核。
- 补抽可能带回同模式漏网项（如 ISCAE 考试中心）→ 补抽结果展示给用户确认后再 build；SKILL 内置指定模式黑名单可扩展。
- 换模型 → **必须删旧 `*.classify_cache.jsonl`**，判定结果不可跨模型混用。
- 换表头 → 零改码（认列 LLM + manifest）。
- 大中断 → 判定缓存续跑，只补中断点之后；finalize 各命令独立无状态。
- 判定失败条目标记 `status=error`，重跑自动重试；全部成功才写判定产物。
- 大批量性能 → 判定是唯一瓶颈（约 2-3s/条），用 `--batch-size 20` + `--workers 10`；语义去重/校验只对抽样后池（100-200 条）做，候选级不做全量两两。
