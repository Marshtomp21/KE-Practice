"""向量化。

提供两种 provider：
- hashing：本地字符 n-gram 哈希向量，零依赖、可离线、结果确定。它保证在没有
  任何外部服务时全流程仍能跑通，也让对比实验的随机性完全可控。
- remote：OpenAI 兼容的 embeddings 接口，配置密钥后自动启用。

两种 provider 输出同样形状的单位向量，检索层不关心用的是哪一种。
"""
from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

import httpx
import numpy as np

from ..core.config import Settings, load_settings

TOKEN_PATTERN = re.compile(r"[一-龥]|[A-Za-z]+|\d+")


class Embedder(ABC):
    """把一批文本变成 (n, dim) 的单位向量矩阵。"""

    dimension: int

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        ...


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class HashingEmbedder(Embedder):
    """字符 unigram + bigram 哈希到固定维度，再做 L2 归一化。

    对中文而言，字与相邻字对已经能提供足够的词面区分度；哈希碰撞用两个不同的
    盐值分散，避免单一哈希把高频字挤在同一维。
    """

    def __init__(self, dimension: int = 512) -> None:
        self.dimension = dimension

    def _tokens(self, text: str) -> List[str]:
        units = TOKEN_PATTERN.findall(text)
        bigrams = [units[i] + units[i + 1] for i in range(len(units) - 1)]
        return units + bigrams

    def _bucket(self, token: str, salt: str) -> int:
        digest = hashlib.md5((salt + token).encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % self.dimension

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in self._tokens(text or ""):
                matrix[row, self._bucket(token, "a")] += 1.0
                matrix[row, self._bucket(token, "b")] += 0.5
        # 次线性压缩高频项，避免长片段仅因为字数多就整体占优
        matrix = np.log1p(matrix)
        return _normalize(matrix)


class RemoteEmbedder(Embedder):
    """调用 OpenAI 兼容的 embeddings 接口。失败时由调用方决定是否降级。"""

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str,
        dimension: int,
        batch_size: int = 32,
        timeout_seconds: float = 60.0,
        trust_env_proxy: bool = False,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.dimension = dimension
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds
        self.trust_env_proxy = trust_env_proxy

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors: List[List[float]] = []
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            with httpx.Client(trust_env=self.trust_env_proxy) as client:
                response = client.post(
                    self.endpoint,
                    json={"model": self.model, "input": batch},
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
            response.raise_for_status()
            body = response.json()
            vectors.extend(item["embedding"] for item in body["data"])
        matrix = np.asarray(vectors, dtype=np.float32)
        self.dimension = matrix.shape[1] if matrix.size else self.dimension
        return _normalize(matrix)


def build_embedder(settings: Optional[Settings] = None) -> Embedder:
    """按配置选 provider；远程所需的密钥缺失时自动退回本地哈希，并且不报错。"""
    settings = settings or load_settings()
    provider = str(settings.get("embedding.provider", "hashing")).lower()
    dimension = int(settings.get("embedding.dimension", 512))
    if provider == "remote":
        api_key = settings.secret("embedding.api_key_env")
        endpoint = settings.get("embedding.endpoint", "")
        model = settings.get("embedding.model", "")
        if api_key and endpoint and model:
            return RemoteEmbedder(
                endpoint=endpoint,
                model=model,
                api_key=api_key,
                dimension=dimension,
                batch_size=int(settings.get("embedding.batch_size", 32)),
                trust_env_proxy=bool(settings.get("embedding.trust_env_proxy", False)),
            )
    return HashingEmbedder(dimension=dimension)
