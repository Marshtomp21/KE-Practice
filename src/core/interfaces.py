"""数据来源、图存储、检索、答案生成与完整问答方法的抽象接口。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .types import (
    Answer,
    Chunk,
    Entity,
    RawDocument,
    Relation,
    RetrievalConstraints,
    RetrievalResult,
)


class DocumentSource(ABC):
    """数据接入层：把某种外部来源转成统一的 RawDocument 流。"""

    @abstractmethod
    def iter_documents(self) -> Iterable[RawDocument]:
        ...

    @abstractmethod
    def describe(self) -> Dict[str, Any]:
        """返回来源的可读描述，用于导入报告。"""


class GraphStore(ABC):
    """图存储层。默认 NetworkX 实现，Neo4j 可作为可选后端接入。"""

    @abstractmethod
    def upsert_entity(self, entity: Entity) -> str:
        ...

    @abstractmethod
    def upsert_relation(self, relation: Relation) -> str:
        ...

    @abstractmethod
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        ...

    @abstractmethod
    def remove_entity(self, entity_id: str) -> bool:
        ...

    @abstractmethod
    def remove_relation(self, relation_id: str) -> bool:
        ...

    @abstractmethod
    def all_entities(self) -> List[Entity]:
        ...

    @abstractmethod
    def all_relations(self) -> List[Relation]:
        ...

    @abstractmethod
    def match_entities(self, text: str, limit: int = 10) -> List[Tuple[Entity, float]]:
        """模糊匹配，返回 (实体, 匹配得分) 且按得分降序。"""

    @abstractmethod
    def neighborhood(
        self, seed_ids: Sequence[str], hops: int = 1, max_nodes: int = 100
    ) -> Tuple[List[Entity], List[Relation]]:
        ...

    @abstractmethod
    def as_networkx(self):
        """导出 networkx 图对象，供 PPR 等图算法直接使用。"""

    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        ...

    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        ...


class Retriever(ABC):
    """检索层统一接口。四种检索方式的唯一差异只体现在 retrieve 的实现里。"""

    name: str = "retriever"

    @abstractmethod
    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        year_range: Optional[Tuple[Optional[int], Optional[int]]] = None,
        constraints: Optional[RetrievalConstraints] = None,
    ) -> RetrievalResult:
        """year_range 为可选的年份窗口；通用检索器可以忽略它。"""


class AnswerGenerator(ABC):
    """生成层：把检索结果组装成带引用的答案。"""

    @abstractmethod
    def generate(self, question: str, result: RetrievalResult) -> Answer:
        ...


class QAMethod(ABC):
    """完整问答方法的统一接口。

    Retriever 适合本地“检索后统一生成”的实验；QAMethod 再向上一层，允许接入
    neo4j-graphrag 这类自行完成检索、增强和生成的库，也作为后续 GraphRAG
    方法复现时的稳定扩展点。
    """

    name: str = "method"

    @abstractmethod
    def ask(
        self,
        question: str,
        top_k: Optional[int] = None,
        constraints: Optional[RetrievalConstraints] = None,
    ) -> Answer:
        ...

    def close(self) -> None:
        """释放数据库连接等可选资源。"""
