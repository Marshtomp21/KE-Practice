"""Enabled QA methods; imports only trigger independent registry entries."""
from .registry import available, build_method, register  # noqa: F401
from . import gap_repair, hipporag2, kg2rag, library_graphrag, vector  # noqa: F401 触发注册

__all__ = ["available", "build_method", "register"]
