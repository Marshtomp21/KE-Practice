# Shared film benchmark

这里保存四套 GraphRAG 复现共同使用的冻结输入、标注规范和统一评分器。

## 当前基准

正式版本为 `benchmarks/l2_film_120_v2/`：

- 120 篇从组内电影数据渲染出的 UTF-8 文档；
- 5 道 dev 题，只用于权重、Prompt 和检索参数调整；
- 20 道冻结 test 题：8 道导演事实题、5 道共同出演路径题、5 道跨影片集合题、2 道无答案题；
- 每题包含 canonical entity ID、aliases、gold documents、精确字符证据跨度和必要关系路径；
- `freeze_manifest.json` 记录 124 个冻结输入文件的 SHA-256。

聚合题先在每部影片内部按规范实体去重，再按“导演 → 不同影片 → 演员实体”统计；
同一演员必须出现在至少两部不同影片中。源数据中张震和毕·彼特的冲突 ID/译名使用
生成器内的显式人工修正规则，不使用不可审计的模糊合并。

## 生成和验证

```bash
python3 scripts/prepare_l2_film_benchmark.py
python3 -m unittest discover -s tests -v
```

`ANNOTATION_REVIEW.md` 是人工复核清单。`freeze_manifest.json` 生成后，测试集内容不得
随调参改变；任何修订都必须提升版本号并重新复核。

## 统一评分

`scripts/benchmark_utils.py` 提供可审计规则：

- 事实/路径题：通过候选实体 aliases 映射到 canonical ID，报告 entity EM、Precision、Recall、F1；
- 列表题：对预测实体 ID 集合计算 Precision、Recall、F1 和 exact match；
- 无答案题：只有答案开头出现明确拒答表达才计为正确；
- 参考文献段落不参与实体命中，且“无法回答……提到正确实体”不会误得分。

已有结果可独立重评分：

```bash
python3 scripts/evaluate_results.py \
  --questions benchmarks/l2_film_120_v2/test.jsonl \
  --input ../SYSTEM/output/RUN/summary.json
```

人工或 LLM judge 只能作为抽样补充；主结果必须保留上述确定性评分明细。

## 目录角色

- `benchmarks/`：版本化、冻结的数据与标注；
- `scripts/`：基准生成、语料审计和统一评分；
- `tests/`：评分规则与冻结数据完整性检查；
- `questions.schema.json`：题目数据契约；
- `corpus_audit.md`：电影源数据审计摘要。
