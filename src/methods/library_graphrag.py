"""基于官方 neo4j-graphrag 包的 GraphRAG 基线。

该方法有意让第三方库负责完整的 retrieval -> augmentation -> generation 流程：
VectorCypherRetriever 从 Neo4j 向量索引命中 Chunk，随后用 Cypher 收集有界邻域子图，
GraphRAG.search 生成最终答案。本模块只做配置、格式转换和资源生命周期管理。
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import httpx

from ..core.config import Settings
from ..core.interfaces import QAMethod
from ..core.types import Answer, Citation, Entity, Relation, Subgraph
from ..generate.answer import EMPTY_REPLY
from ..retrieve.embedding import build_embedder
from .registry import register


class MethodUnavailable(RuntimeError):
    """可选依赖或外部服务未配置。"""


def _api_base_url(endpoint: str) -> str:
    suffix = "/chat/completions"
    endpoint = (endpoint or "").rstrip("/")
    return endpoint[: -len(suffix)] if endpoint.endswith(suffix) else endpoint


@register("library_graphrag")
class LibraryGraphRAGMethod(QAMethod):
    """`neo4j-graphrag` 的 VectorCypherRetriever + GraphRAG 适配器。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.default_top_k = int(settings.get("retrieval.top_k_chunks", 6))
        self._driver: Any = None
        self._rag: Any = None
        self._http_client: Optional[httpx.Client] = None

    def _ensure_runtime(self) -> None:
        if self._rag is not None:
            return
        try:
            from neo4j import GraphDatabase
            from neo4j_graphrag.embeddings.base import Embedder as Neo4jEmbedder
            from neo4j_graphrag.generation import GraphRAG, RagTemplate
            from neo4j_graphrag.llm import OpenAILLM
            from neo4j_graphrag.retrievers import VectorCypherRetriever
            from neo4j_graphrag.types import RetrieverResultItem
        except ImportError as exc:
            raise MethodUnavailable(
                "library_graphrag 需要可选依赖：pip install -r requirements-graphrag.txt"
            ) from exc

        uri = self.settings.secret("library_graphrag.neo4j.uri_env")
        user = self.settings.secret("library_graphrag.neo4j.user_env")
        password = self.settings.secret("library_graphrag.neo4j.password_env")
        api_key = self.settings.secret("llm.api_key_env")
        if not uri or not user or not password:
            raise MethodUnavailable("请配置 NEO4J_URI、NEO4J_USER 和 NEO4J_PASSWORD")
        if not api_key:
            raise MethodUnavailable("library_graphrag 需要配置 GRAPHRAG_LLM_KEY")

        project_embedder = build_embedder(self.settings)

        class ProjectEmbedder(Neo4jEmbedder):
            def embed_query(self, text: str) -> List[float]:
                return project_embedder.encode([text])[0].astype(float).tolist()

        # neo4j-graphrag 1.19 仍调用兼容的 queryNodes 过程。Aura 会为此返回一条
        # DEPRECATION 通知；关闭通知不影响查询错误/异常，只避免控制台打印整段 Cypher。
        self._driver = GraphDatabase.driver(
            uri, auth=(user, password), notifications_min_severity="OFF"
        )
        try:
            self._driver.verify_connectivity()
            retriever = VectorCypherRetriever(
                driver=self._driver,
                index_name=str(self.settings.get("library_graphrag.index_name", "text_embeddings")),
                embedder=ProjectEmbedder(),
                retrieval_query=self._retrieval_query(
                    int(self.settings.get("library_graphrag.max_relations_per_chunk", 20))
                ),
                result_formatter=self._result_formatter(RetrieverResultItem),
                neo4j_database=self.settings.get("library_graphrag.neo4j.database") or None,
            )
            self._http_client = httpx.Client(
                trust_env=bool(self.settings.get("llm.trust_env_proxy", False))
            )
            llm = OpenAILLM(
                model_name=str(self.settings.get("llm.model", "")),
                api_key=api_key,
                base_url=_api_base_url(str(self.settings.get("llm.endpoint", ""))),
                model_params={"temperature": float(self.settings.get("generation.temperature", 0.0))},
                http_client=self._http_client,
            )
            template = RagTemplate(
                template=(
                    "请仅根据给定上下文回答问题；上下文不足时明确说明，不要补充外部事实。\n\n"
                    "问题：\n{query_text}\n\n上下文：\n{context}\n\n回答："
                ),
                expected_inputs=["query_text", "context"],
            )
            self._rag = GraphRAG(retriever=retriever, llm=llm, prompt_template=template)
        except Exception:
            self.close()
            raise

    @staticmethod
    def _retrieval_query(max_relations_per_chunk: int = 20) -> str:
        """收集有界的一跳邻域，避免高连接度节点造成路径枚举爆炸。"""
        relation_limit = max(1, min(int(max_relations_per_chunk), 100))
        return f"""
        WITH node AS chunk, score
        OPTIONAL MATCH (chunk)<-[:FROM_CHUNK]-(seed:Entity)
        CALL (seed) {{
            OPTIONAL MATCH (seed)-[relation:KG_RELATION]-(neighbor:Entity)
            WITH relation, neighbor
            ORDER BY coalesce(relation.collaboration_count, 0) DESC,
                     elementId(relation)
            LIMIT {relation_limit}
            RETURN collect(relation) AS relation_list,
                   collect(neighbor) AS neighbor_list
        }}
        WITH chunk, score,
             [n IN [seed] + neighbor_list WHERE n IS NOT NULL |
                {{id: elementId(n), name: coalesce(n.name, n.title, ''),
                 type: coalesce(n.type, head(labels(n)))}}
             ] AS nodes,
             [r IN relation_list WHERE r IS NOT NULL |
                {{id: elementId(r), source: elementId(startNode(r)),
                 target: elementId(endNode(r)), type: coalesce(r.type, type(r))}}
             ] AS edges
        RETURN {{
            text: coalesce(chunk.text, ''),
            chunk_id: coalesce(chunk.id, elementId(chunk)),
            score: score,
            nodes: nodes,
            edges: edges
        }} AS result
        """

    @staticmethod
    def _result_formatter(item_type: Any):
        def format_record(record: Any):
            payload = dict(record["result"] or {})
            nodes = payload.get("nodes") or []
            edges = payload.get("edges") or []
            names = {node.get("id"): node.get("name", "") for node in nodes}
            triples = [
                f"{names.get(edge.get('source'), edge.get('source', ''))} "
                f"-[{edge.get('type', '')}]-> "
                f"{names.get(edge.get('target'), edge.get('target', ''))}"
                for edge in edges
            ]
            text = str(payload.get("text", ""))
            context = text
            if triples:
                context += "\n相关图谱关系：\n" + "\n".join(triples)
            return item_type(content=context, metadata=payload)

        return format_record

    def ask(self, question: str, top_k: Optional[int] = None) -> Answer:
        started = time.perf_counter()
        self._ensure_runtime()
        limit = top_k or self.default_top_k
        response = self._rag.search(
            query_text=question,
            retriever_config={"top_k": limit},
            return_context=True,
            response_fallback=EMPTY_REPLY,
        )
        items = list(getattr(getattr(response, "retriever_result", None), "items", []) or [])
        citations, subgraph = self._convert_context(items)
        return Answer(
            text=str(response.answer),
            citations=citations,
            subgraph=subgraph,
            retriever_name=self.name,
            latency=time.perf_counter() - started,
            debug_info={
                "method": self.name,
                "backend": "neo4j-graphrag",
                "top_k": limit,
                "context_items": len(items),
            },
        )

    @staticmethod
    def _convert_context(items: List[Any]) -> tuple[List[Citation], Subgraph]:
        citations: List[Citation] = []
        nodes: Dict[str, Entity] = {}
        edges: Dict[str, Relation] = {}
        scores: Dict[str, float] = {}
        for index, item in enumerate(items, start=1):
            metadata: Dict[str, Any] = dict(getattr(item, "metadata", {}) or {})
            text = str(metadata.get("text") or getattr(item, "content", ""))
            chunk_id = str(metadata.get("chunk_id") or f"neo4j-context-{index}")
            citations.append(Citation(
                marker=f"S{index}", doc_id="neo4j", chunk_id=chunk_id,
                char_start=0, char_end=len(text), snippet=text,
            ))
            score = float(metadata.get("score") or 0.0)
            for raw in metadata.get("nodes") or []:
                node_id = str(raw.get("id", ""))
                if not node_id:
                    continue
                nodes.setdefault(node_id, Entity(
                    id=node_id, name=str(raw.get("name") or node_id),
                    type=str(raw.get("type") or "Entity"),
                ))
                scores[node_id] = max(scores.get(node_id, 0.0), score)
            for raw in metadata.get("edges") or []:
                edge_id = str(raw.get("id", ""))
                if edge_id:
                    edges.setdefault(edge_id, Relation(
                        id=edge_id,
                        head_id=str(raw.get("source", "")),
                        tail_id=str(raw.get("target", "")),
                        type=str(raw.get("type") or "RELATED_TO"),
                    ))
        highlight = [key for key, _ in sorted(scores.items(), key=lambda pair: -pair[1])[:6]]
        return citations, Subgraph(
            entities=list(nodes.values()), relations=list(edges.values()),
            node_scores=scores, highlight_path=highlight,
        )

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
        if self._http_client is not None:
            self._http_client.close()
        self._driver = None
        self._rag = None
        self._http_client = None
