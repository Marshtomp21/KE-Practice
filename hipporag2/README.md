# HippoRAG 2 复现

本目录使用官方 OpenIE、短语/同义关系图和 personalized PageRank 完成电影问答。官方版本固定为 `OSU-NLP-Group/HippoRAG@eb0568d6f75bac037b37e7404603462db60ffac2`。

适配器为 DeepSeek LLM 和 SiliconFlow Embedding 注入独立客户端，不改动 upstream；每题保存 answer、Top documents、scores、graph seeds、metadata 和 latency。

## 环境

```bash
git clone https://github.com/OSU-NLP-Group/HippoRAG.git upstream
git -C upstream checkout eb0568d6f75bac037b37e7404603462db60ffac2
python3.12 -m venv ../.venv-hipporag2
../.venv-hipporag2/bin/pip install -e upstream
cp config/api.env.example config/api.env
```

## 运行

```bash
../.venv-hipporag2/bin/python scripts/demo_dual_api.py \
  --l2 --output-dir output/l2_film_120_v2 --dry-run
../.venv-hipporag2/bin/python scripts/demo_dual_api.py \
  --l2 --output-dir output/l2_film_120_v2 --allow-api
```

最终结果位于 `output/l2_film_120_v2/summary.json`。本次索引包含 2,589 个节点和 8,461 条边记录，mean F1 1.0000、exact match 1.0000。
