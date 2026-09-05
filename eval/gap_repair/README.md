# GapRepair：查询时缺边补偿

这是按照《知识图谱缺边补偿 Benchmark 实验报告》实现的项目级研究方法。
核心问题是：向量检索得到的文本证据，能否恢复不完备图上的查询路径。
不把“问题分解 + 自适应检索”本身宣称为首次提出的新方法。

当前默认是增强版：候选边界实体优先查询、命中文档局部扩展、分层证据以及确定性
集合计算。自动生成的角色片单单独标为 `dataset_assertion`（置信度 0.65），
不冒充自然语言直接蕴含的高可信关系。`corpus_records: false` 可关闭这层解释。

## 使用

```bash
python scripts/ask.py "王俊凯与苗苗共同出演了哪些电影？" --retriever gap_repair --show-debug
```

Web 方法列表也提供 GapRepair。API 返回的 `graph.nodes` 包含标为“本次问答证据”
的 EvidenceNode，并用临时 `supports` 边连接相关实体；原始图关系与临时证据分开保存。
能否回答具体问题取决于本地图与 Chunk 索引是否包含相关原始材料。

## 实现机制

1. **关系约束计划**：仅从问题文本与实体词表识别共同出演、全部导演、导演—演员
   影片交集、两导演影片中的共同演员、同导演不同影片的重复演员。别名按词表匹配；
   同名 `NAME:`/`WP:` 标识仅在有唯一同类型 Wikidata ID 时归一化。没有匹配的题型
   回退 HippoRAG2 检索与共享生成器，不冒充已经完成缺口推理。
2. **可见图扩展与核查**：屏蔽边只在视图边界过滤。规划、验证、答案计算均不读取
   `masked_edges` 的内容或 `supplemental_queries`。支持物理删除边与查询屏蔽的等价测试。
   先核查最多 40 个图关系链接的 Chunk；已有完整路径不等于集合答案已经完整。
3. **定向检索**：将未知 Movie/Person 槽位转成“人物 + 关系 + 影片”、
   “已知候选影片 + 待核查人物 + 关系”等查询；默认最多两轮、32 个补偿查询、
   每次取 20 个 Chunk。增强版总查询预算为 32，优先查询只缺一侧支持的演员，
   并沿命中文档补查其角色片单；最终输出仍最多 6 个 Chunk。
   恢复第一跳后继续展开可见图第二跳。中间检索预算高于单次向量基线，不能视为等成本比较。
4. **关系验证**：只接受明确角色谓词与本地片名绑定，支持人物传记的有限主语省略。
   保存逐字可回查的 `char_start`、`char_end`、`raw_text`。拒绝只共现、否定、
   假设选角、跨影片歧义以及“参演或关联”这类不能确定关系类型的描述。
   这是影视领域的轻量规则验证器，召回有限；不是通用语言理解或离线全图抽取。
   另一层可选证据只解析导入器生成的完整角色片单格式，依据“角色对应的片单成员”
   假设推导数据集内的关系，保留较弱可信度与假设字段。它不是原始维基文本的语义蕴含；
   新版实验必须同时报告关闭此层的分数。
5. **临时证据图**：修复边只活在本次调用内，与来源 Chunk 一同形成 EvidenceNode。
   不调用持久图写入、源数据写入或索引持久化接口。下一次问题重新建立查询视图。
6. **预算内证明选择**：先枚举满足问题约束的候选证明，再贪心选择可复用原文、
   新增 Chunk 少的证明组合。默认最终最多 6 个 Chunk；缺少所选文本支持的临时边
   不进入答案。字符预算再次截断时同步撤回失去引用的答案。
   这是近似的紧凑证据图选择，**不保证数学意义上的全局最小或全集完整性**。
7. **确定性答案**：直接在证据图上执行交集及按影片去重后的计数，列出实体与引用。
   `gap_repair.answer_mode: llm` 可切换共享生成器，并将有来源支持的临时关系传给它。
   比较历史 LLM 方法时必须注明生成机制不同，不能把全部差值归因于检索。

## 缺口与无答案的边界

`provisional_gaps` / `gap_suspected` 是检索前的核查需求，含 `no_complete_proof`、
`text_graph_disagreement`、`set_completeness_unverified`。后者只是集合完整性尚未证实。
`gap_detected` 指核查后产生的、可见图未包含的关系，属于后验诊断。启用片单策略时，
其中也包含依据角色片单假设得到的推断关系，必须结合 `evidence_tiers` 判断证据强弱。
它不能被报告成“检索前缺口检测准确率”。

仅有不完整图和有限 top-k 文本，通常无法证明一个共同答案绝不存在。默认先做补偿核查；
对于明确限定“当前语料”的是否题，只有双方片单均被检索、未找到共同证明且引用预算
保留双方材料时，才回答“交叉核查后，未发现共同出演的影片（限本次检索到的证据）”。
这是有范围的检索观察，`negative_assessment.exhaustive=false`，**不是证明不存在**。
否则仍返回“证据不足／无法确定”。评分器接受“未发现”的否定措辞，应公开此评价边界。
集合核查还会在完整图触发补偿，无谓补偿率可能偏高。当前实现没有解决低误触发率问题。

## 评测与消融

通用 runner 已支持注册方法及独立配置：

```bash
python eval/run_benchmark_v2.py --methods gap_repair --split dev --graph-view masked --settings config/settings.yaml --result-dir eval/results/gap_repair_standard
```

另提供自动 complete/masked 配对、代码和索引摘要绑定、断点续跑的入口：

```bash
python eval/run_gap_repair.py --split dev
python eval/run_gap_repair.py --split test
python eval/run_gap_repair.py --split test --ablation no_compensation
python eval/run_gap_repair.py --split test --ablation no_repair
python eval/run_gap_repair.py --split test --no-corpus-records --workers 4
python eval/run_gap_repair.py --split test --answer-mode llm --graph-view masked --workers 4
```

还支持 `--ablation always`、`--ablation no_prune`、`--top-k`；`--offline` 会在内存中
重建独立 hashing 索引，不覆盖原 BGE 索引，离线分数不能和 BGE 正式分数混报。
各配置结果写入独立签名目录，题集标注只传给屏蔽视图和评分器，不传给算法。
查询缓存只缓存与图视图无关的向量命中，绝不缓存修复图；支持线程并发评测。

本次工作区的源数据已被用户扩充，与冻结题集的 SHA256 不符。因此实际评测使用
从提交 `c58f966` 导出并恢复 LF 换行的独立源数据：

```bash
python eval/run_gap_repair.py --split test --source-dir eval/results/gap_repair_frozen/data/source/wikipedia_300_films_final
```

该目录在本次工作区已准备好，未覆盖用户源数据。迁移机器时应从冻结提交重新导出，
并确保四个源 JSONL 的组合摘要与 `questions.yaml` 一致；不要用 `--allow-source-drift`
将变化后的语料当作正式复现。已有索引保留原样，实验签名记录其 SHA256。

消融含义：`no_compensation` 保留相同规划、证明选择与确定性生成，关闭所有文本补边；
`no_repair` 仍执行补偿查询，但禁止临时关系进入推理。后者的确定性生成器不会直接
从未验证文本推导答案，因此用于测试“显式修复关系”的作用，不等价于原报告的 LLM naive_hybrid。

## 代码与验证

- `src/retrieve/gap_plan.py`：查询语法和集合证明。
- `src/retrieve/gap_evidence.py`：关系验证及字符级溯源。
- `src/retrieve/gap_repair.py`：查询视图、补偿循环、证明预算。
- `src/methods/gap_repair.py`：生成、引用与临时子图输出。
- `tests/test_gap_repair.py`：双边缺失、物理删除等价、不读 oracle、不写回图、
  共现和否定、片名误绑定、别名、去重计数、预算、API 临时节点等回归验证。

实际测试结果见同目录 `RESULTS.md`。未来值得优先改进的是带置信度的关系验证器、
跨文档人物主语解析，以及在开放世界下区分真实负例与证据不足的方法。
