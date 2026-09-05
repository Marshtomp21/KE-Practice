"""检索装配上下文；完整问答方法在 src.methods.registry 注册。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from ..core.config import Settings, load_settings
from ..core.interfaces import GraphStore
from ..core.types import Chunk
from .anchors import AnchorResolver
from .vector_index import ChunkVectorIndex

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
