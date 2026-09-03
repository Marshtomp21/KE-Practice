"""问答方法注册表。

运行时只面向 QAMethod，不关心一种方法内部是本地检索后生成，还是第三方库提供
的完整 GraphRAG pipeline。新增方法只需实现接口并使用 @register 注册。
"""
from __future__ import annotations

from typing import Callable, Dict, List, Type

from ..core.config import Settings
from ..core.interfaces import QAMethod

_REGISTRY: Dict[str, Type[QAMethod]] = {}


def register(name: str) -> Callable[[Type[QAMethod]], Type[QAMethod]]:
    def decorate(cls: Type[QAMethod]) -> Type[QAMethod]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorate


def available() -> List[str]:
    return sorted(_REGISTRY)


def build_method(name: str, settings: Settings) -> QAMethod:
    try:
        cls = _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"未注册的问答方法 {name!r}，可用的有 {available()}") from exc
    return cls(settings)
