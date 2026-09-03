# 影视领域 GraphRAG 方法实验基座

项目当前聚焦检索问答方法本身，只保留两个可运行基线：

- `vector`：项目内的本地向量 RAG。使用 NumPy 向量索引召回文本，由统一生成器回答。
- `library_graphrag`：调用官方 `neo4j-graphrag` 包的 `VectorCypherRetriever` 和
  `GraphRAG`。先从 Neo4j 向量索引命中 Chunk，再扩展两跳子图并生成答案。

文档清洗、知识抽取和图谱构建代码仍保留为离线基础设施，但不再进入默认运行主线。
后续复现新的 GraphRAG 方法时，实现 `QAMethod` 并注册即可，CLI、API、前端和评测
入口都不需要改。

## 当前架构

```text
                         ┌─ vector ─ 本地 NPZ 向量索引 ─ 统一生成器
问题 ─ QAService ─ 方法注册表
                         └─ library_graphrag ─ Neo4j VectorCypherRetriever
                                                └─ GraphRAG.search()
```

主要代码：

```text
src/methods/
  registry.py             完整问答方法注册表
  vector.py               本地向量 RAG 基线
  library_graphrag.py     官方库 GraphRAG 适配器
src/generate/service.py   CLI / API / 评测共用入口
src/retrieve/             底层向量工具及暂不启用的旧实验检索器
scripts/build_index.py    只构建本地向量基线
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
python scripts/make_questions.py
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

## 4. Web 与评测

```bash
python scripts/serve.py
```

浏览器访问 `http://127.0.0.1:8000/`，可以切换两个方法。

只测试本地向量基线：

```bash
python eval/run_compare.py
```

Neo4j 和 LLM 均已配置时比较两个方法：

```bash
python eval/run_compare.py --retrievers vector,library_graphrag
```

## 5. 添加后续 GraphRAG 方法

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

## 6. 测试

```bash
python -m pytest -q
```

`legacy/` 仅作为历史参考，不参与测试收集或当前运行。
