# Film GraphRAG Gap Benchmark v2（40 题）

该 benchmark 研究一个明确问题：**当知识图谱缺少关键关系，但原始 Chunk 仍保留
相关事实时，检索方法能否发现证据缺口，并用文本证据恢复多跳推理？**

题集不会修改持久图。runner 会为每道题创建只读查询视图，在检索时临时过滤指定
关系；原始 Chunk、向量索引和完整图文件保持不变。因而同一道题可以在 `complete`
和 `masked` 两种图视图下配对测试，测量缺边导致的性能下降及文本补偿带来的恢复。

## 题集结构

| 场景 | 数量 | 含义 |
|---|---:|---|
| `critical_edge_missing` | 22 | 隐藏答案路径上的关键边，文本证据仍存在 |
| `count_support_missing` | 8 | 隐藏跨影片计数所需的部分出演边 |
| `complete_control` | 4 | 不隐藏边，校准多答案完整性 |
| `negative_control` | 6 | 本来就无共同答案，检验误关联和无谓补偿 |

题型包括 4 道多导演、8 道共同出演影片、8 道导演演员交集、8 道跨影片重复演员、
6 道导演—演员影片交集和 6 道困难负例。共 40 题，其中 `dev` 8 题用于调试，
`test` 32 题保持冻结，只用于最终报告。

每题包含 canonical entity ID、aliases、gold documents、字符级 evidence、必要关系
路径，以及 `graph_perturbation`：屏蔽边、补偿证据文档和 oracle 查询。`source_sha256`
绑定四个源数据文件，避免语料变化后误用旧真值。

## 比较方法

- `vector`：只检索 Chunk，不依赖图完整性。
- `library_graphrag`：Neo4j 官方库的向量起点与 Cypher 图扩展。
- `kg2rag`：语义 Chunk 种子、有界图扩展与联合重排。
- `hipporag2`：实体种子、PPR 图传播和桥接路径保留。
- `naive_hybrid`：实验对照；不检测缺口，每题都合并 HippoRAG2 图结果与原问题的
  向量检索结果。它用于判断提升是否只来自“多检索一次”。
- `oracle_repair`：性能上界；直接使用人工标注的 gold 补偿查询检索 Chunk，并把有
  gold 文档支持的屏蔽关系作为本次查询的临时关系。它不是可部署方法，也不参与
  “谁最好”的公平排名，用于估计补偿模块仍有多少可提升空间。

后续自己的方法应在不知道 gold 的前提下输出 `gap_detected`、
`compensation_triggered`、`compensation_documents`，以及有文本支撑的
`temporary_relations`，即可使用同一评分器。

## 评分规则

- 回答：别名归一化后的实体集合 Precision/Recall/F1/Exact Match；额外答案会降低
  Precision。无答案题必须明确否定，“材料不足”不算正确。
- 防泄漏：先从模型输出中剔除原文转录与证据附录，再抽取答案实体。
- 检索：gold document recall/F1、关系 edge recall、可见图 path coverage。
- 缺口修复：gap detection accuracy、补偿 document recall、无谓补偿率，以及加入
  受 gold 文档支持的临时关系后的 recovered path coverage。
- 保留旧式 gold 子串召回仅作诊断，不作为主指标。

难度来自多跳交集、集合完整性、跨文档计数、强干扰负例和受控缺边，而不是故意
制造含糊题目。题目不会根据一次跑分自动替换，也不以“必须压低某方法得分”为
标注目标。

## 构建与静态校验

`questions.yaml` 是冻结交付文件。仅在源语料或题目规格明确变化时重新生成：

```bash
python scripts/build_benchmark_v2.py
python -m pytest -q tests/test_benchmark_v2.py tests/test_qa_methods.py
```

## 实验流程

先在 dev 集做完整图/缺边图配对测试：

```bash
python eval/run_benchmark_v2.py --split dev --graph-view complete
python eval/run_benchmark_v2.py --split dev --graph-view masked
```

配置冻结后再运行 test：

```bash
python eval/run_benchmark_v2.py --split test --graph-view complete
python eval/run_benchmark_v2.py --split test --graph-view masked
python eval/summarize_benchmark_v2.py
```

默认比较六种方法。若 Neo4j 尚未可用，可用 `--methods` 暂时排除
`library_graphrag`。结果分别写入 `eval/results/benchmark_v2/complete/` 和
`eval/results/benchmark_v2/masked/`；恢复运行会按“题目 × 方法 × graph view”跳过
已完成项，`--fresh` 只会重建当前 graph view 的结果文件。
