# Microsoft GraphRAG 复现

本目录复现 Microsoft GraphRAG 的图构建、Leiden 社区划分、社区报告和 Global Search。官方版本固定为 `microsoft/graphrag@243637c4eb94e34c3a5e5c7d871a725e8d6b77fc`（`v3.1.2`）。

DeepSeek 使用 JSON-object 响应格式；`build_community_reports_deepseek.py` 将经过校验的对象转换为官方 `community_reports.parquet` 契约。适配器逐社区 checkpoint，并保持 upstream 不变。

## 环境

```bash
git clone https://github.com/microsoft/graphrag.git upstream
git -C upstream checkout 243637c4eb94e34c3a5e5c7d871a725e8d6b77fc
python3.12 -m venv ../.venv-microsoft-graphrag
../.venv-microsoft-graphrag/bin/pip install graphrag==3.1.2
```

运行脚本读取 `../kg2rag/config/api.env`，将 LLM 与 Embedding 凭证映射到子进程环境。

## 运行

```bash
../.venv-microsoft-graphrag/bin/python scripts/prepare_l2_workspace.py
../.venv-microsoft-graphrag/bin/python scripts/run_official_index.py --dry-run
../.venv-microsoft-graphrag/bin/python scripts/run_official_index.py --allow-api
../.venv-microsoft-graphrag/bin/python scripts/build_community_reports_deepseek.py \
  --workspace workspace_l2_v2 --allow-api
../.venv-microsoft-graphrag/bin/python scripts/run_l2_global_eval.py --dry-run
../.venv-microsoft-graphrag/bin/python scripts/run_l2_global_eval.py --allow-api
```

最终查询结果为 `output/l2_film_120_v2_global.json`，精简索引统计为 `output/l2_film_120_v2_index_summary.json`。本次构建 1,797 个实体、2,006 条关系和 271 份社区报告；test mean F1 0.8121、exact match 0.7000。
