"""当前启用的问答方法：本地向量基线与官方库 GraphRAG 基线。"""
from .registry import available, build_method, register  # noqa: F401
from . import library_graphrag, vector  # noqa: F401 触发注册

__all__ = ["available", "build_method", "register"]
