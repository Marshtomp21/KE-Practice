# LightRAG 复现

本目录使用 LightRAG hybrid 模式完成实体/关系抽取、低层与高层检索、原文 Chunk 融合和回答生成。官方版本固定为 `HKUDS/LightRAG@28ff1b05f2ac3f3e6fa14dd2cd33656579bd0c9c`（`v1.5.7`）。

## 环境

```bash
git clone https://github.com/HKUDS/LightRAG.git upstream
git -C upstream checkout 28ff1b05f2ac3f3e6fa14dd2cd33656579bd0c9c
python3.12 -m venv ../.venv-lightrag
../.venv-lightrag/bin/pip install -e upstream
```

运行器读取 `../kg2rag/config/api.env`，不会复制或输出密钥。

## 运行

```bash
../.venv-lightrag/bin/python scripts/run_l2_film_api.py --dry-run
../.venv-lightrag/bin/python scripts/run_l2_film_api.py --allow-api --mode hybrid
```

完整索引保存在本地 `output/l2_film_120_v2/storage/`，Git 只保留精简结果 `summary.json`。本次图包含 2,073 个节点和 2,286 条边，mean F1 1.0000、exact match 1.0000。
