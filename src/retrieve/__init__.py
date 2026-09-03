"""底层检索工具。

当前运行主线由 src.methods 注册完整问答方法。本目录中旧的 traverse/ppr/hybrid
实验实现保留供参考，但不再自动导入或暴露给服务。
"""
from .embedding import Embedder, build_embedder
from .vector_index import ChunkVectorIndex

__all__ = ["Embedder", "build_embedder", "ChunkVectorIndex"]
