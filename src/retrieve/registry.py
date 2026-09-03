"""检索器注册表与装配上下文。

检索器通过 @register 把自己登记进表里，调用方只用配置里的名字取实例。
这样"新增一种检索方式"只需要新写一个类，任何调用点都不需要改动，也就不会
出现按检索类型分支的 if/else。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Type

from ..core.config import Settings, load_settings
from ..core.interfaces import GraphStore, Retriever
from ..core.types import Chunk
from .anchors import AnchorResolver
from .vector_index import ChunkVectorIndex

_REGISTRY: Dict[str, Type[Retriever]] = {}


def register(name: str) -> Callable[[Type[Retriever]], Type[Retriever]]:
    def decorate(cls: Type[Retriever]) -> Type[Retriever]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorate


def available() -> List[str]:
    return sorted(_REGISTRY)


@dataclass
class RetrievalContext:
    """所有检索器共享的资源。共享同一份索引与同一个锚点解析器是公平性前提。"""

    store: GraphStore
    index: ChunkVectorIndex
    settings: Settings
    anchors: AnchorResolver

    @classmethod
    def assemble(
        cls,
        store: GraphStore,
        index: ChunkVectorIndex,
        settings: Optional[Settings] = None,
    ) -> "RetrievalContext":
        settings = settings or load_settings()
        return cls(
            store=store,
            index=index,
            settings=settings,
            anchors=AnchorResolver(
                store, min_score=float(settings.get("retrieval.anchor_min_score", 0.55))
            ),
        )

    def chunk_lookup(self) -> Dict[str, Chunk]:
        return {chunk.id: chunk for chunk in self.index.chunks}


def build_retriever(name: str, context: RetrievalContext) -> Retriever:
    """按名字取检索器。名字来自 settings.yaml 或前端下拉框。"""
    try:
        cls = _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"未注册的检索器 {name!r}，可用的有 {available()}") from exc
    return cls(context)
