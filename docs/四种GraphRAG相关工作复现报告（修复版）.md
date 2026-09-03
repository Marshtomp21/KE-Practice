# 四种 GraphRAG 相关工作复现与课程项目建议

> 课程设计：基于知识图谱的 RAG 系统构建  
> 复现对象：KG²RAG、HippoRAG 2、LightRAG、Microsoft GraphRAG  
> 统一基准：`l2_film_120_v2`  
> 实验日期：2026-09-04

## 1. 工作概述

本阶段围绕电影领域知识问答，完成了四种代表性 GraphRAG 方法的方法级复现，并建立了一套统一、可审计的比较流程。具体工作包括：

- 阅读课程要求、学校 baseline、电影数据集和当前项目基座；
- 固定四个官方项目的代码版本，并使用相互隔离的运行环境；
- 将四种方法适配到相同的中文电影语料、LLM、Embedding 模型和问题集；
- 构建包含规范实体、别名、证据文档、字符级证据和关系路径的冻结基准；
- 为不同系统实现可恢复的索引、查询和结果保存脚本；
- 使用统一规则计算事实题、路径题、集合题和无答案题指标；
- 分析四种方法对课程项目可直接采用的模块，并给出后续实现路线。

最终，四种方法都完成了 120 篇电影文档的索引与 20 道冻结测试题。KG²RAG、HippoRAG 2 和 LightRAG 的 mean F1 与 exact match 均为 1.0000；Microsoft GraphRAG Global Search 的 mean F1 为 0.8121、exact match 为 0.7000。这里的分数用于验证课程语料上的方法流程与工程闭环，不等同于论文原数据集上的指标复刻。

## 2. 课程项目背景

### 2.1 课程要求与当前选题

课程设计要求系统至少包含知识图谱构建、检索器和大语言模型三个核心组件，同时需要明确实体与关系的抽取方式、图检索方式以及知识注入方式。项目还应具备可运行的系统展示、版本管理、测试和质量检查，并体现小组自己的设计。

当前选题为电影领域 GraphRAG 问答系统。该方向能够同时覆盖：

- 影片、导演、演员、角色、公司、类型、奖项等实体与关系；
- 导演查询、演员查询等单跳事实问题；
- 共同出演、导演—影片—演员等多跳问题；
- 跨影片演员统计和合作网络等集合问题；
- 电影主题、创作群体和类型趋势等全局综合问题；
- 证据不足时的拒答和来源追踪。

### 2.2 学校 baseline

学校提供的 [`code/2026graphrag`](../../2026graphrag/) 以《红楼梦》为示例，包含 LLM 实体关系抽取、Neo4j 写入、Chunk 向量索引、向量/混合/图增强检索、问答生成和 Web 展示。它为课程项目提供了完整的组件组合参考。

本项目在此基础上重点加强以下方面：

- 将领域逻辑从固定示例中解耦，使用电影 Schema 和配置文件；
- 将检索方法抽象为统一接口，便于公平比较和消融；
- 对图扩展规模、证据数量和 Token 预算进行约束；
- 在回答中保留文档、实体、路径、分数和证据跨度；
- 建立独立的 dev/test 评测协议。

### 2.3 电影数据集

[`code/KE-Practice-film-data`](../../KE-Practice-film-data/) 当前包含：

| 数据项 | 数量 |
|---|---:|
| 影片 | 361 |
| 演员 | 1,784 |
| 导演 | 274 |
| 显式关系 | 4,941 |
| 派生合作关系 | 14,610 |
| 含完整中文维基源文本和导言的影片 | 361 |
| 含独立人物正文的演员 | 1,164 |
| 含独立人物正文的导演 | 194 |

这批数据事实关系丰富，适合检验实体链接、跨文档关系发现、路径检索、图传播和集合聚合。其主体仍是百科知识，而不是稳定标注了作者、时间、观点和论据的专业影评；因此现阶段更准确的定位是“电影知识与评价相关语料上的 GraphRAG 问答”。若后续强化影评分析，应补充多来源评论及观点级标注。

### 2.4 当前项目基座

[`code/KE-Practice`](../../KE-Practice/) 已具备模块化的课程项目骨架：

- YAML 管理模型、路径、切块、Schema 和抽取规则；
- `QAMethod` 注册机制隔离不同问答方法；
- 本地向量索引和 Neo4j `VectorCypherRetriever`；
- PPR、图遍历和混合检索的基础实现；
- 统一的回答、引用、子图、路径、分数和调试信息结构；
- CLI、FastAPI、Web 图谱与问答界面；
- 数据构建、抽取评测和问答比较脚本。

相关工作复现的作用不是把四个大型框架整体并入基座，而是选择其中可解释、可消融的模块，接入现有统一接口。

## 3. 复现范围与统一配置

### 3.1 固定版本

| 方法 | 官方仓库 | 固定版本 |
|---|---|---|
| KG²RAG | `nju-websoft/KG2RAG` | `7d626c77b7af30b55aa3f960cde755b9549a0616` |
| HippoRAG 2 | `OSU-NLP-Group/HippoRAG` | `eb0568d6f75bac037b37e7404603462db60ffac2` |
| LightRAG | `HKUDS/LightRAG` | `v1.5.7` / `28ff1b05f2ac3f3e6fa14dd2cd33656579bd0c9c` |
| Microsoft GraphRAG | `microsoft/graphrag` | `v3.1.2` / `243637c4eb94e34c3a5e5c7d871a725e8d6b77fc` |

官方仓库按上述 commit 获取，适配代码单独放在各系统的 `scripts/` 中，不直接改动官方源码。四套系统使用独立 Python 环境，API Key 仅从本地未跟踪的配置读取。

### 3.2 模型与服务

- 生成和信息抽取：`deepseek-v4-flash`，关闭 thinking；
- 向量表示：`BAAI/bge-m3`；
- KG²RAG 二次排序：`BAAI/bge-reranker-v2-m3`；
- DeepSeek 与 SiliconFlow 分别使用独立的 OpenAI-compatible 客户端；
- 远程运行必须显式传入 `--allow-api`，并可先用 `--dry-run` 检查输入和调用规模。

该配置保持了各方法的主要检索流程，但替换了论文或官方示例中的本地模型。因此本工作属于统一电影语料上的方法级/API 适配复现。

### 3.3 官方最小流程验证

在运行共享电影基准前，四个系统均完成了最小流程验证：KG²RAG 走通种子检索、图扩展、重排和生成；HippoRAG 2 完成文档 OpenIE、图构建、PPR 和问答；LightRAG 完成文档插入、实体关系图构建和 local 查询；Microsoft GraphRAG 完成实体关系抽取、社区报告和 Global Search。随后再使用完全相同的冻结电影输入执行 L2 实验。

## 4. 统一电影评测基准

### 4.1 数据划分

冻结基准位于 `shared/benchmarks/l2_film_120_v2/`：

- 120 篇由组内电影数据渲染的 UTF-8 文档；
- dev 5 题：事实题 2、路径题 1、集合题 1、无答案题 1；
- test 20 题：事实题 8、路径题 5、集合题 5、无答案题 2；
- `manifest.jsonl` 固定文档 ID、路径和内容哈希；
- `freeze_manifest.json` 固定语料、dev、test 和复核文件的 SHA-256；
- `ANNOTATION_REVIEW.md` 保存人工复核记录。

dev 只用于接口、Prompt 和检索配置检查，test 用于最终统计。完成 test 后不再基于其答案调整检索权重和 Prompt。

### 4.2 标注内容

每道题保存：

- `canonical_id`、规范名称和 aliases；
- `gold_documents`；
- `gold_evidence` 及 `char_start` / `char_end`；
- `required_relation_paths`；
- 参与精确评分的 `answer_candidates`；
- 是否可回答及答案类型。

集合题先在单部影片内按规范实体去重，再按“导演 → 不同影片 → 演员集合”统计。演员必须出现在同一导演的至少两部不同影片中。张震、毕·彼特、杰森·薛兹曼等多译名实体通过显式规则归一到同一 canonical ID。

### 4.3 统一评分

`shared/scripts/benchmark_utils.py` 提供确定性评分：

- 事实题与路径题：别名归一化后的实体 Precision、Recall、F1 和 exact match；
- 集合题：规范实体 ID 集合的 Precision、Recall、F1 和 exact match；
- 无答案题：显式拒答准确率；
- References 不参与实体匹配；
- 带解释的列表答案优先提取结论、总结句或答案列表，再进行集合评分。

评分器和冻结基准共有 13 个自动测试；KG²RAG 另有 4 个 OpenIE JSON/XML 解析测试。最终全部通过。

## 5. 四种方法的复现过程与结果

### 5.1 KG²RAG

#### 实现流程

适配器实现了以下流程：

```text
问题向量化
  → 语义种子文档
  → OpenIE 知识图谱邻域扩展
  → 候选文档二次排序
  → 证据组织与答案生成
```

为保证构图和查询过程可审计，实现中加入：

- 每篇文档、每次 OpenIE 尝试的原始响应；
- JSON、fenced JSON、XML 字段式和 XML 简写式解析；
- 解析重试和逐文档 checkpoint；
- 文档覆盖率、非空文档比例和三元组数量质量门；
- 种子文档、向量分数、扩展文档数量、重排证据、回答和延迟记录；
- dev/test 分离及按题型运行能力。

#### 实验结果

| 指标 | 结果 |
|---|---:|
| 文档 | 120 |
| OpenIE 原始响应覆盖 | 120/120 |
| 非空三元组文档 | 119/120（99.17%） |
| 三元组 | 2,471 |
| 端到端首次构建记录 | 约 332.115 s |
| 平均查询时间 | 3.265 s |
| test mean F1 | 1.0000 |
| test exact match | 1.0000 |

#### 对项目的价值

KG²RAG 的“语义种子—图扩展—重排”结构适合直接作为课程项目的主检索骨架：实现成本适中，每一步都能解释，并且能够在现有 Neo4j 图和 Chunk 向量索引上完成，不需要引入整套外部框架。

### 5.2 HippoRAG 2

#### 实现流程

使用官方 HippoRAG 2 完成 OpenIE、短语图构建、同义边构建、personalized PageRank 文档排序和 RAG QA。适配器分别注入 DeepSeek LLM 与 SiliconFlow Embedding 客户端，并按官方接口解包：

```python
query_solutions, response_messages, all_metadata = rag.rag_qa(...)
```

每道题单独保存 answer、Top documents、scores、graph seeds、response message、metadata 和 latency。上游没有本电影数据集专用 Prompt，本实验使用其 MUSIQUE 多跳 QA Prompt，并在报告中明确记录这一配置。

#### 实验结果

| 指标 | 结果 |
|---|---:|
| 文档 / passage nodes | 120 / 120 |
| phrase nodes | 2,469 |
| 总节点 | 2,589 |
| OpenIE 三元组 | 3,305 |
| passage 关联三元组 | 3,166 |
| 同义边 | 2,211 |
| 图中总边记录 | 8,461 |
| 端到端首次构建记录 | 584.033 s |
| 平均查询时间 | 2.522 s |
| test mean F1 | 1.0000 |
| test exact match | 1.0000 |

#### 对项目的价值

最值得采用的是以查询实体相关度作为 personalization、在实体—关系—Chunk 图上传播分数的 PPR 思路。课程项目可实现轻量版 PPR，并增加高度节点惩罚、传播范围限制和路径回溯，使跨影片、多跳和别名关系检索更稳定、更可解释。

### 5.3 LightRAG

#### 实现流程

固定官方 LightRAG `v1.5.7`，使用 hybrid 模式完成：

- 文档切块和实体关系抽取；
- 实体、关系、Chunk 向量存储；
- 低层实体关键词与高层关系关键词检索；
- local/global 信息融合；
- 证据上下文组织和生成。

运行脚本复用官方持久化存储和缓存，最终仓库只保留精简结果；完整索引可以通过同一命令重新生成。

#### 实验结果

| 指标 | 结果 |
|---|---:|
| 文档 | 120 |
| 图节点 | 2,073 |
| 图边 | 2,286 |
| 端到端首次构建记录 | 450.000 s |
| 平均查询时间 | 3.042 s |
| test mean F1 | 1.0000 |
| test exact match | 1.0000 |

#### 对项目的价值

LightRAG 的关键价值是双层查询表示：低层关键词面向具体实体和事实，高层关键词面向关系和主题。课程项目可以借鉴这一思想做问题路由和检索融合，同时采用其增量插入与持久化设计支持知识更新演示。

### 5.4 Microsoft GraphRAG

#### 实现流程

使用官方 `v3.1.2` 完成文档切分、实体关系抽取、描述归并、Leiden 社区划分和 Global Search。针对 DeepSeek 的响应格式能力，外置兼容适配器以 JSON-object 模式生成社区报告，执行字段校验后写入官方 `community_reports.parquet` 数据契约。适配器具有逐社区 checkpoint 和重试能力，官方源码保持不变。

Global Search 对所选层级社区报告执行 map-reduce，并逐题保存 answer、score、latency、CLI 返回码和标准错误信息。

#### 实验结果

| 指标 | 结果 |
|---|---:|
| 文档 / text units | 120 / 120 |
| 实体 | 1,797 |
| 关系 | 2,006 |
| 社区 / 社区报告 | 271 / 271 |
| 索引与社区报告时间 | 约 1,070.711 s |
| 平均查询时间 | 5.660 s |
| test mean F1 | 0.8121 |
| test exact match | 0.7000 |

分题型结果：

| 题型 | Mean F1 |
|---|---:|
| 事实题 | 0.7500 |
| 路径题 | 0.9333 |
| 集合题 | 0.7152 |
| 无答案题 | 1.0000 |

#### 对项目的价值

社区摘要和 map-reduce 适合回答跨大量文档的主题、群体和趋势问题，但社区压缩会减少演员表等细粒度事实。课程项目应将 Global Search 作为全局问题的专用路径，与实体级 Local/Basic Search 配合，而不是让所有问题统一经过社区摘要。

## 6. 结果比较与解释

| 方法 | 图/索引规模 | 冷索引时间 | 平均查询 | Test F1 | EM |
|---|---|---:|---:|---:|---:|
| KG²RAG | 2,471 triples | ≈332.115 s | 3.265 s | 1.0000 | 1.0000 |
| HippoRAG 2 | 2,589 nodes / 8,461 edge records | 584.033 s | 2.522 s | 1.0000 | 1.0000 |
| LightRAG | 2,073 nodes / 2,286 edges | 450.000 s | 3.042 s | 1.0000 | 1.0000 |
| Microsoft GraphRAG Global | 1,797 entities / 2,006 relations / 271 reports | ≈1,070.711 s | 5.660 s | 0.8121 | 0.7000 |

时间来自单次本机端到端运行，包含外部 API 波动，只用于观察本实验的工程成本。KG²RAG 与 HippoRAG 2 支持 checkpoint；其最终 test JSON 中的索引字段记录恢复运行耗时，表格使用首次完整构建记录。前三种方法在当前小规模测试集上均为满分，说明它们能够覆盖这些显式电影事实，不足以证明方法性能等价。Microsoft Global Search 的题型差异则说明：检索路径需要随问题粒度变化。

下一阶段应加入更有区分度的问题，包括 3 跳以上推理、别名冲突、缺边、干扰证据、多来源观点、时间约束和全局主题总结，并通过消融实验验证各模块的真实贡献。

## 7. 建议采用的项目方案

### 7.1 模块选择

| 可采用设计 | 来源 | 项目中的用途 | 建议 |
|---|---|---|---|
| 语义种子 Chunk | KG²RAG | 保证初始证据与问题相关 | 直接采用 |
| 有界图扩展 | KG²RAG | 补充跨 Chunk 关系证据 | 直接采用 |
| 图引导重排 | KG²RAG | 控制噪声和 Token 预算 | 直接采用 |
| Personalized PageRank | HippoRAG 2 | 多跳传播和跨文档聚合 | 实现轻量版 |
| 同义/别名边 | HippoRAG 2 | 中文译名和实体对齐 | 直接采用 |
| 低层/高层查询表示 | LightRAG | 区分事实词与主题词 | 直接采用 |
| 增量索引与缓存 | LightRAG | 知识更新和工程演示 | 作为系统亮点 |
| 社区摘要 | Microsoft GraphRAG | 全局主题与群体问题 | 按需采用 |
| Map-reduce 全局问答 | Microsoft GraphRAG | 大范围语料综合 | 只用于全局路径 |
| 全链路检索审计 | 本次统一评测 | 展示证据、路径和分数 | 必须采用 |

### 7.2 推荐总体架构

建议把课程项目收敛为“问题路由 + 分层证据图检索 + 可追溯生成”：

```text
用户问题
  → 实体链接与别名归一化
  → 问题类型路由
      ├─ 单事实：向量种子 + 实体邻居
      ├─ 多跳/集合：种子 + 有界扩展 + PPR + 重排
      └─ 全局主题：社区摘要 + map-reduce
  → Evidence 组装与 Token 预算控制
  → 受证据约束的答案生成
  → 答案、引用、路径、分数、耗时统一返回
```

项目创新点可以落在：

1. **自适应问题路由**：根据实体数、关系词、集合词、全局主题词和检索置信度选择路径；
2. **受约束图扩展**：综合语义相关度、路径长度、边类型和高度节点惩罚控制扩展；
3. **证据图组织**：将 Chunk、三元组和关系路径按支持链排列，并给出字符级引用；
4. **检索升级机制**：局部证据覆盖不足时，再升级到 PPR 或社区摘要。

## 8. 后续评测与验收指标

### 8.1 检索层

- Gold document Recall@K；
- Gold evidence span Recall@K；
- Relation/path recall；
- 扩展节点数、候选 Chunk 数和噪声比例；
- 不同题型的路由准确率。

### 8.2 生成层

- 单实体 EM/F1；
- 集合 Precision/Recall/F1；
- 无答案题准确率；
- 引用正确性与答案忠实度；
- 人工或 LLM judge 抽样复核，并保留原始判断记录。

### 8.3 工程层

- 索引时间、查询时间和 Token/API 成本；
- 图规模、文档覆盖率和非空关系比例；
- checkpoint、重试和缓存命中情况；
- 结果的可复现性和审计字段完整性。

近期验收建议：

- 四种外部方法均可从冻结输入重新运行并产生统一格式结果；
- 当前项目至少提供 Vector、KG²RAG-style、PPR 和路由式混合方法；
- 至少展示一个需要图路径的多跳案例和一个全局社区摘要案例；
- 所有回答可回溯到文档证据和关系路径；
- 在 hidden test 上完成主实验和关键模块消融；
- Web 页面能够展示答案、引用、子图和检索轨迹。

## 9. 下一步计划

### P0：接入课程项目基座

1. 统一 `Document / Entity / Relation / Evidence / RetrievalTrace` 数据结构；
2. 将 KG²RAG-style seed-expand-rerank 接入 `QAMethod`；
3. 将现有 PPR 实现升级为可配置、可解释的正式方法；
4. 增加规则式问题路由器，并为后续学习式路由保留接口；
5. 在调试接口和前端展示种子、扩展路径、重排分数和引用。

### P1：建立公平实验轨道

1. 内部轨道统一 Chunk、Embedding、LLM 和候选预算，比较 Vector、固定邻域、KG²RAG-style、PPR 和路由方法；
2. 外部轨道保留四种官方系统各自的原生索引和检索流程，作为系统级参考；
3. 对问题路由、图扩展、PPR、重排和社区摘要分别做消融；
4. 每次实验保存配置、版本、结果和检索 trace。

### P2：扩展数据和题型

1. 新建不公开答案的 hidden test，避免继续围绕当前 test 调整；
2. 增加多来源评论，标注评论者、时间、观点对象、立场和论据；
3. 增加冲突证据、时间约束、拼写噪声、缺边和无答案问题；
4. 增加真正需要社区摘要的全局比较与趋势问题。

## 10. 代码与产物

仓库只保留复现所需代码、冻结输入、精简审计数据和最终结果；虚拟环境、官方仓库克隆、模型缓存和可再生索引不上传。

- 总说明：`code/reproductions/README.md`
- 冻结基准：`code/reproductions/shared/benchmarks/l2_film_120_v2/`
- 统一评分器：`code/reproductions/shared/scripts/benchmark_utils.py`
- KG²RAG 结果：`code/reproductions/kg2rag/output/l2_film_120_v2/summary.json`
- HippoRAG 2 结果：`code/reproductions/hipporag2/output/l2_film_120_v2/summary.json`
- LightRAG 结果：`code/reproductions/lightrag/output/l2_film_120_v2/summary.json`
- Microsoft GraphRAG 结果：`code/reproductions/microsoft_graphrag/output/l2_film_120_v2_global.json`
- Microsoft GraphRAG 索引摘要：`code/reproductions/microsoft_graphrag/output/l2_film_120_v2_index_summary.json`

本报告完整记录了本阶段采用的方法、统一实验协议、实现方式、最终结果、可迁移模块和下一步计划，可作为相关工作复现阶段的主要交付文档。
