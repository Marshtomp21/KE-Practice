# 影视领域 GraphRAG 方法实验基座

GapRepair 使用影视领域定制的查询计划与证据验证规则，保留片单补偿、查询级临时 EvidenceNode，以及确定性输出或共享 LLM 输出。详见 [方法说明](eval/gap_repair/README.md)。

项目当前聚焦检索问答方法本身，提供两个基线、两个相关工作方法和查询级缺边补偿方法：

- `vector`：项目内的本地向量 RAG。使用 NumPy 向量索引召回文本，由统一生成器回答。
- `kg2rag`：KG²RAG-style 方法。语义检索产生种子 Chunk，执行有界知识图扩展，
  再用语义相关度与图证据支持度联合重排。
- `hipporag2`：HippoRAG 2-style 方法。以问题实体作为重启分布，在实体关系图上
  执行带高度节点惩罚的 Personalized PageRank，并保留查询实体间的桥接路径。
- `library_graphrag`：调用官方 `neo4j-graphrag` 包的 `VectorCypherRetriever` 和
  `GraphRAG`。先从 Neo4j 向量索引命中 Chunk，再扩展两跳子图并生成答案。
- `gap_repair`：将影视问题转为关系约束，在可见图上检查证据缺口，定向检索 Chunk，
  校验原文关系后构建临时 EvidenceNode，并在引用预算内执行集合交集和去重计数。
  不训练模型、不写回知识图谱。参见 [方法说明与实验](eval/gap_repair/README.md)。

保留文档导入、清洗、切分和索引构建；本地图由结构化电影数据只读加载。
旧检索器、离线知识抽取、合成语料及旧评测流程已移除。
后续复现新的 GraphRAG 方法时，实现 `QAMethod` 并注册即可，CLI、API、前端和评测
入口都不需要改。

## 当前架构

```text
                         ┌─ vector ─ 本地 NPZ 向量索引 ──────────────┐
                         ├─ kg2rag ─ 语义种子 ─ 图扩展 ─ 重排 ──────┤
问题 ─ QAService ─ 方法注册表
                         ├─ hipporag2 ─ 实体种子 ─ PPR ─ 路径证据 ─┤─ 回答
                         ├─ gap_repair ─ 缺口检测 ─ 临时证据补偿 ────┤
                         └─ library_graphrag ─ Neo4j GraphRAG ──────┘
```

主要代码：

```text
src/methods/
  registry.py             完整问答方法注册表
  gap_repair.py           电影领域缺边补偿方法适配器
  vector.py               本地向量 RAG 基线
  kg2rag.py               KG²RAG-style 方法适配器
  hipporag2.py            HippoRAG 2-style 方法适配器
  library_graphrag.py     官方库 GraphRAG 适配器
src/generate/service.py   CLI / API / 评测共用入口
src/retrieve/kg2rag.py     种子、扩展与候选重排
src/retrieve/hipporag2.py  高度惩罚 PPR 与桥接路径
src/retrieve/dataset_graph.py  结构化电影关系到本地图的只读适配
scripts/build_index.py    构建本地方法共用的 Chunk 向量索引
scripts/sync_neo4j.py     一次性同步现有数据到 Neo4j
```

## 1. 安装

核心依赖：

```bash
pip install -r requirements.txt
```

如果要运行官方库 GraphRAG：

```bash
pip install -r requirements-graphrag.txt
```

## 2. 本地向量 RAG

从 `data/source/wikipedia_300_films_final/` 导入真实的影片、演员和导演语料：

```bash
python scripts/import_wikipedia_films.py --clean
```

构建本地索引：

```bash
python scripts/build_index.py
```

已有 `data/interim/chunks.jsonl` 时可直接复用：

```bash
python scripts/build_index.py --skip-ingest
```

问答：

```bash
python scripts/ask.py "《卧虎藏龙》是谁导演的？" --retriever vector
```

当前统一使用 SiliconFlow 的 `BAAI/bge-m3` 生成 1024 维向量，本地向量基线与
Neo4j GraphRAG 共用同一批 embedding 和同一个查询编码器。密钥由 `.env` 中的
`GRAPHRAG_LLM_KEY` 提供。

## 3. 官方库 GraphRAG

复制环境变量模板：

```bash
copy .env.example .env
```

至少填写：

```dotenv
GRAPHRAG_LLM_KEY=...
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
```

将已有本地 Chunk、向量以及真实数据集自带的关系增量同步到 Neo4j：

```bash
python scripts/sync_neo4j.py
```

更换 embedding 模型或维度后，重建向量索引：

```bash
python scripts/sync_neo4j.py --recreate-index
```

同步脚本使用 `MERGE`，不会清空数据库。它创建/复用配置中的 `text_embeddings`
向量索引。Neo4j 索引维度必须与 `config/settings.yaml` 的
`embedding.dimension` 一致。

运行库方法：

```bash
python scripts/ask.py "梁朝伟与章子怡通过哪些影片产生关联？" --retriever library_graphrag --show-debug
```

这里确实调用 `neo4j_graphrag.generation.GraphRAG.search()`；库负责检索、上下文增强
和答案生成，本项目只负责配置以及把库结果转换成统一的 `Answer`。

## 4. KG²RAG 与 HippoRAG 2

两个方法共享 `vector` 的 Chunk、Embedding 和生成器，以保证方法对比时只改变检索
过程。知识图在方法首次使用时从 `data/source/wikipedia_300_films_final/` 的结构化
关系只读构建到内存，不修改 Neo4j，也不写入其他方法的索引。

完成第 2 节的数据导入和向量索引构建后即可运行：

```bash
python scripts/ask.py "王俊凯与苗苗通过哪部影片产生关联？" \
  --retriever kg2rag --show-debug

python scripts/ask.py "哪些演员多次出现在宫崎骏执导的影片中？" \
  --retriever hipporag2 --show-debug
```

两种方法的参数分别位于 `config/settings.yaml` 的 `kg2rag` 和 `hipporag2` 配置块。
调试结果包含语义种子、图种子、扩展规模、PPR 排名、桥接路径、候选分项得分和
最终重排结果，便于前端展示与消融实验。

## 5. Web 与评测

```bash
python scripts/serve.py
```

浏览器访问 `http://127.0.0.1:8000/`，可以切换五种问答方法。
示例问题读取 Benchmark v2 开发集；页面仅做交互演示，不会应用评测缺边掩码。
正式方法比较使用下方的 Benchmark v2 与 [GapRepair 评测入口](eval/gap_repair/README.md)。

面向“不完备知识图谱检索”的 40 题 benchmark 位于
`eval/benchmark_v2/questions.yaml`。其中 30 题会在单次查询的只读图视图中隐藏
关键关系，但保留原始 Chunk 证据；另有 4 题完整图校准和 6 题困难负例。题集包含
8 道 dev 与 32 道冻结 test，详细设计和评分口径见 `eval/benchmark_v2/README.md`。

benchmark 还提供两个仅用于实验的对照：`naive_hybrid` 每题无条件合并图与向量
检索，`oracle_repair` 使用 gold 补偿查询作为理想信息对照，而非数学意义的性能上界。它们不会注册到 Web 或正式
方法列表，不影响其他成员继续开发方法。

建议先对 dev 集运行完整图/缺边图配对实验：

```bash
python eval/run_benchmark_v2.py --split dev --graph-view complete
python eval/run_benchmark_v2.py --split dev --graph-view masked
```

## 6. 添加后续 GraphRAG 方法

新方法实现：

```python
from src.core.interfaces import QAMethod
from src.methods.registry import register

@register("my_method")
class MyMethod(QAMethod):
    def __init__(self, settings):
        self.settings = settings

    def ask(self, question, top_k=None):
        ...  # 返回 src.core.types.Answer
```

然后在 `src/methods/__init__.py` 导入模块触发注册。方法可以复用当前的 Chunk、
Neo4j 图、自己的索引，或者调用第三方实现；上层始终只接收统一 `Answer`。

## 7. 测试

```bash
python -m pytest -q
```
