"""统一问答服务。

暴露本地向量基线、两个相关工作方法和 neo4j-graphrag 官方库实现。调用方始终
面向 QAMethod，因此各方法的资源与检索过程互相隔离。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..core.config import Settings, load_settings
from ..core.interfaces import QAMethod
from ..core.types import Answer, RetrievalConstraints
from ..methods import available, build_method


class QAService:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or load_settings()
        self.default_retriever = str(self.settings.get("qa.default_method", "vector"))
        self._methods: Dict[str, QAMethod] = {}

    @property
    def retriever_names(self) -> List[str]:
        """保留旧 API 字段名，避免前端和评测脚本同时发生无意义变更。"""
        return available()

    def method(self, name: Optional[str] = None) -> QAMethod:
        key = name or self.default_retriever
        if key not in self.retriever_names:
            raise KeyError(f"未知问答方法 {key!r}，可用的有 {self.retriever_names}")
        if key not in self._methods:
            self._methods[key] = build_method(key, self.settings)
        return self._methods[key]

    def ask(
        self,
        question: str,
        retriever_name: Optional[str] = None,
        top_k: Optional[int] = None,
        year_range=None,
        constraints: Optional[RetrievalConstraints] = None,
    ) -> Answer:
        # year_range 仅为兼容旧调用签名；当前完整方法不实现时间过滤。
        return self.method(retriever_name).ask(
            question, top_k=top_k, constraints=constraints
        )

    def graph_stats(self) -> Dict[str, object]:
        embedding_file = self.settings.path("paths.embedding_file")
        return {
            "local_vector_index": embedding_file.exists(),
            "local_vector_index_path": str(embedding_file),
            "dataset_graph_source": self.settings.path("paths.dataset_dir").exists(),
            "note": "library_graphrag 的图谱与向量索引位于 Neo4j",
        }

    def close(self) -> None:
        for method in self._methods.values():
            method.close()
        self._methods.clear()
