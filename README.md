# kb-query-pipeline | 知识库 query 筛选抽样流水线

知识库问答评测数据的预处理流水线：用 LLM 识别列映射、判定"指定 query"与质量问题，确定性剔除 + 分层随机抽样 + 转 zhfmt 格式，抽样后人工复核闭环。换表头零改码，断点续跑，seed 可复现。

> 配套工具：评测打分用 [kb-qa-eval](internal-gitlab/eval_project)；报告生成用 [kb-report](internal-gitlab/eval_report)。

## 快速开始

```powershell
$env:KB_QA_API_KEY = "<key>"

# 自动链：认列 → 判定 → 剔除 → 分层抽样 → 转 zhfmt
python scripts/pipeline.py --file <检索结果.xlsx> --n 100 --schema-yes

# 人工复核闭环
python scripts/finalize.py dedup --pool <池.xlsx> --out-dir <数据目录>
```

完整五阶段流程、参数表、判定口径、人工复核闭环、已知边界见 [SKILL.md](SKILL.md)。

## 依赖

- Python venv：共享 kb-qa-eval 的 venv（`pandas` / `openpyxl` / `requests`）
- 环境变量：
  - `KB_QA_API_KEY`（必填）
  - `KB_QA_API_BASE_URL`（可选，默认 `https://maragate.aibotplatform.com/v1`）
  - `KB_QA_MODEL`（可选，默认 `azure/gpt-5.4`；**换模型 = 删旧 `*.classify_cache.jsonl` 重新判定**）

## 项目结构

```
scripts/
  pipeline.py     # 自动链：认列→判定→剔除→抽样→转zhfmt
  finalize.py     # 人工复核闭环：dedup/refill/build/final-check
  prompts.py      # schema detection 与判定 prompt（语言无关，含黄金回归用例）
tests/
  test_pipeline.py / test_finalize.py / test_cache.py
  fixtures/make_data.py / run_tests.py
SKILL.md          # Agent skill 定义（含完整判定口径与已知边界）
```

## 测试

```powershell
python tests/run_tests.py
```