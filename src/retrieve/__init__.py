"""底层检索工具。

运行主线由 ``src.methods`` 注册完整问答方法。KG²RAG 与 HippoRAG 2 的检索器由
各自方法直接装配；traverse/ppr/hybrid 实验实现保留参考，但不自动注册到服务。
"""
from .embedding import Embedder, build_embedder
from .vector_index import ChunkVectorIndex

__all__ = ["Embedder", "build_embedder", "ChunkVectorIndex"]
