# KG²RAG 复现

本目录实现电影语料上的“语义种子检索 → KG 邻域扩展 → 文档重排 → 回答生成”。官方版本固定为 `nju-websoft/KG2RAG@7d626c77b7af30b55aa3f960cde755b9549a0616`，适配代码位于 `scripts/`。

## 环境

```bash
git clone https://github.com/nju-websoft/KG2RAG.git upstream
git -C upstream checkout 7d626c77b7af30b55aa3f960cde755b9549a0616
python3.9 -m venv ../.venv-kg2rag
../.venv-kg2rag/bin/pip install -r config/requirements-compatible.txt
cp config/api.env.example config/api.env
```

`config/api.env` 只保存在本地。默认使用 DeepSeek 生成/OpenIE、BGE-M3 Embedding 和 BGE reranker。

## 运行

```bash
../.venv-kg2rag/bin/python -m unittest discover -s tests -v
../.venv-kg2rag/bin/python scripts/run_l2_film_api.py --dry-run
../.venv-kg2rag/bin/python scripts/run_l2_film_api.py --allow-api
```

运行器保存每篇文档的原始 OpenIE 响应、JSON/XML 解析结果、种子文档、扩展规模、重排证据、逐题答案与延迟。查询开始前要求：原始响应覆盖率 100%、非空文档比例至少 80%、三元组至少 120。

最终结果位于 `output/l2_film_120_v2/summary.json`；本次得到 2,471 条三元组、mean F1 1.0000、exact match 1.0000。
