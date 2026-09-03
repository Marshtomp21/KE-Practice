# GraphRAG 相关工作复现

本仓库在统一中文电影基准上复现 KG²RAG、HippoRAG 2、LightRAG 和 Microsoft GraphRAG，用于知识工程课程项目的方法选型与模块设计。正式总结见 [`docs/四种GraphRAG相关工作复现报告（修复版）.md`](docs/四种GraphRAG相关工作复现报告（修复版）.md)。

## 最终结果

| 方法 | 文档 | Test Mean F1 | Exact Match | 平均查询时间 |
|---|---:|---:|---:|---:|
| KG²RAG | 120 | 1.0000 | 1.0000 | 3.265 s |
| HippoRAG 2 | 120 | 1.0000 | 1.0000 | 2.522 s |
| LightRAG | 120 | 1.0000 | 1.0000 | 3.042 s |
| Microsoft GraphRAG Global | 120 | 0.8121 | 0.7000 | 5.660 s |

这些结果验证的是统一电影语料上的方法流程，不是论文原数据集指标复刻。完整索引时间、图规模、分题型结果和方法分析见正式报告。

## 仓库结构

```text
reproductions/
├── docs/                         # 唯一正式复现报告
├── shared/
│   ├── benchmarks/l2_film_120_v2/  # 120 文档、dev 5、test 20
│   ├── scripts/                 # 基准生成、审计、统一评分
│   └── tests/                   # 冻结数据和评分器测试
├── kg2rag/                      # KG²RAG API 适配、测试和审计结果
├── hipporag2/                   # HippoRAG 2 双提供方适配和结果
├── lightrag/                    # LightRAG hybrid 适配和结果
└── microsoft_graphrag/          # GraphRAG 索引/社区报告/Global Search 适配
```

仓库包含适配代码、冻结输入、审计数据和最终结果。官方仓库克隆、虚拟环境、模型缓存和可再生索引不纳入版本管理，可按照下述固定版本和运行说明构建。

## 环境构建与完整运行

建议按以下顺序复现实验：

1. 按下一节的仓库地址和 commit，将官方项目克隆到对应目录的 `upstream/`；
2. 按各系统 README 的“环境”部分创建独立虚拟环境并安装依赖；
3. 从 `api.env.example` 创建本地 `api.env`，填写 DeepSeek 和 SiliconFlow 凭证；
4. 先执行各系统的 `--dry-run` 检查输入范围，再使用新的输出目录或 workspace 完整运行。

```bash
# KG²RAG：执行 OpenIE、构图、检索与评测
.venv-kg2rag/bin/python kg2rag/scripts/run_l2_film_api.py \
  --output-dir kg2rag/output/run_l2_film_120 --allow-api

# HippoRAG 2：从文档建立索引并完成评测
.venv-hipporag2/bin/python hipporag2/scripts/demo_dual_api.py \
  --l2 --force-reindex --output-dir hipporag2/output/run_l2_film_120 --allow-api

# LightRAG：建立持久化索引并执行 hybrid 查询
.venv-lightrag/bin/python lightrag/scripts/run_l2_film_api.py \
  --output-dir lightrag/output/run_l2_film_120 --mode hybrid --allow-api

# Microsoft GraphRAG：创建独立 workspace
.venv-microsoft-graphrag/bin/python microsoft_graphrag/scripts/prepare_l2_workspace.py \
  --workspace microsoft_graphrag/workspace_run_l2_film_120
```

Microsoft GraphRAG 后续索引、社区报告和 Global Search 命令见其目录 README；这些命令均支持通过 `--workspace` 指向新建的 workspace。完整运行会将 120 篇电影文本发送到已配置的 DeepSeek 和 SiliconFlow 服务，并产生相应 API 调用成本。

## 固定的官方版本

| 目录 | 官方仓库 | Commit |
|---|---|---|
| `kg2rag/` | <https://github.com/nju-websoft/KG2RAG.git> | `7d626c77b7af30b55aa3f960cde755b9549a0616` |
| `hipporag2/` | <https://github.com/OSU-NLP-Group/HippoRAG.git> | `eb0568d6f75bac037b37e7404603462db60ffac2` |
| `lightrag/` | <https://github.com/HKUDS/LightRAG.git> | `28ff1b05f2ac3f3e6fa14dd2cd33656579bd0c9c` (`v1.5.7`) |
| `microsoft_graphrag/` | <https://github.com/microsoft/graphrag.git> | `243637c4eb94e34c3a5e5c7d871a725e8d6b77fc` (`v3.1.2`) |

克隆时将仓库放到相应系统的 `upstream/`，再 checkout 表中 commit。适配代码不修改 upstream。

## 统一基准与评分

冻结测试集为 `shared/benchmarks/l2_film_120_v2/test.jsonl`，参数和 Prompt 只允许使用 `dev.jsonl` 调整。每题包含 canonical entity ID、aliases、gold documents、字符级 evidence span 和必要关系路径。

评分规则：

- 事实/路径题：别名归一化后的实体 EM、Precision、Recall、F1；
- 集合题：规范实体集合 Precision、Recall、F1 和 exact match；
- 无答案题：显式拒答准确率；
- References 与解释性证据不作为最终预测实体。

生成并验证基准：

```bash
python3 shared/scripts/prepare_l2_film_benchmark.py
python3 -m unittest discover -s shared/tests -v
```

离线重评分：

```bash
python3 shared/scripts/evaluate_results.py \
  --questions shared/benchmarks/l2_film_120_v2/test.jsonl \
  --input SYSTEM/output/RESULT.json
```

## 环境与凭证

建议在本目录分别创建：

```text
.venv-kg2rag/
.venv-hipporag2/
.venv-lightrag/
.venv-microsoft-graphrag/
```

将 `kg2rag/config/api.env.example` 复制为 `kg2rag/config/api.env` 并填写本地凭证。该文件及所有 `api.env`、`.env` 均被忽略。默认实验使用：

- DeepSeek `deepseek-v4-flash`；
- SiliconFlow `BAAI/bge-m3`；
- KG²RAG 使用 `BAAI/bge-reranker-v2-m3`。

四套 L2 运行都会将冻结电影文本发送给配置的外部服务，必须在获得数据外发授权后使用 `--allow-api`。先运行各系统 README 中的 `--dry-run` 命令检查范围。

## 最终产物

- `kg2rag/output/l2_film_120_v2/summary.json`
- `hipporag2/output/l2_film_120_v2/summary.json`
- `lightrag/output/l2_film_120_v2/summary.json`
- `microsoft_graphrag/output/l2_film_120_v2_global.json`
- `microsoft_graphrag/output/l2_film_120_v2_index_summary.json`

结果 JSON 保存逐题回答、指标和延迟；KG²RAG 额外保留逐文档 OpenIE 原始响应与解析三元组。大体积向量库和图索引不纳入 Git，可由运行脚本重新构建。
