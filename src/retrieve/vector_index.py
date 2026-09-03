"""片段向量索引：一个 npz 文件 + 一次矩阵乘法。

语料规模在几千段量级，暴力余弦检索毫秒级即可返回，引入向量数据库只会增加
演示现场的故障面。索引与片段文本一起存盘，加载后即可独立使用。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..core.config import Settings, load_settings
from ..core.types import Chunk
from .embedding import Embedder, build_embedder


class ChunkVectorIndex:
    """维护 chunk_id -> 向量 的映射，并提供 top-k 余弦检索。"""

    def __init__(
        self, settings: Optional[Settings] = None, embedder: Optional[Embedder] = None
    ) -> None:
        self.settings = settings or load_settings()
        self.embedder = embedder or build_embedder(self.settings)
        self._matrix: Optional[np.ndarray] = None
        self._chunks: List[Chunk] = []

    @property
    def dimension(self) -> int:
        return self._matrix.shape[1] if self._matrix is not None else self.embedder.dimension

    @property
    def size(self) -> int:
        return len(self._chunks)

    def build(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        texts = [f"{c.metadata.get('title', '')}\n{c.text}" for c in self._chunks]
        self._matrix = self.embedder.encode(texts) if texts else np.zeros((0, self.embedder.dimension), dtype=np.float32)

    def persist(self) -> str:
        target = self.settings.path("paths.embedding_file")
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            matrix=self._matrix if self._matrix is not None else np.zeros((0, 1), dtype=np.float32),
            chunks=np.asarray(
                [json.dumps(c.to_dict(), ensure_ascii=False) for c in self._chunks], dtype=object
            ),
        )
        return str(target)

    def load(self) -> "ChunkVectorIndex":
        target = self.settings.path("paths.embedding_file")
        if not target.exists():
            raise FileNotFoundError(f"向量索引不存在: {target}")
        payload = np.load(target, allow_pickle=True)
        self._matrix = payload["matrix"].astype(np.float32)
        self._chunks = [Chunk.from_dict(json.loads(item)) for item in payload["chunks"]]
        return self

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        if self._matrix is None or not self._chunks:
            return []
        query_vector = self.embedder.encode([query])
        if query_vector.shape[1] != self._matrix.shape[1]:
            raise ValueError("查询向量维度与索引不一致，请重建索引")
        scores = self._matrix @ query_vector[0]
        top_k = max(1, min(top_k, len(self._chunks)))
        order = np.argpartition(-scores, top_k - 1)[:top_k]
        order = order[np.argsort(-scores[order])]
        return [(self._chunks[i], float(scores[i])) for i in order if scores[i] > 0]

    def chunks_by_id(self, chunk_ids: Sequence[str]) -> List[Chunk]:
        wanted = set(chunk_ids)
        return [chunk for chunk in self._chunks if chunk.id in wanted]

    @property
    def chunks(self) -> List[Chunk]:
        return list(self._chunks)

    @property
    def vectors(self) -> np.ndarray:
        """返回只读用途的向量矩阵；供 Neo4j 同步脚本复用同一批 embedding。"""
        if self._matrix is None:
            return np.zeros((0, self.embedder.dimension), dtype=np.float32)
        return self._matrix.copy()
