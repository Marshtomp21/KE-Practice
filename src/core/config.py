"""配置与本体加载。

settings.yaml 用点号路径读取（settings.get("hipporag2.alpha")），
schema.yaml 加载为 SchemaRegistry，负责关系头尾类型的合法性判定。
两者都在进程内缓存，避免重复读盘。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_FILE = PROJECT_ROOT / "config" / "settings.yaml"
load_dotenv(PROJECT_ROOT / ".env", override=False)


class ConfigError(Exception):
    """配置或本体文件不合法。"""


class Settings:
    """settings.yaml 的只读视图。"""

    def __init__(self, data: Dict[str, Any], source: Path) -> None:
        self._data = data
        self.source = source

    def get(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted_key: str) -> Any:
        value = self.get(dotted_key, _MISSING)
        if value is _MISSING:
            raise ConfigError(f"settings.yaml 缺少必需项: {dotted_key}")
        return value

    def path(self, dotted_key: str) -> Path:
        """路径项一律相对项目根目录解析，脚本从任何工作目录启动都不受影响。"""
        raw = self.require(dotted_key)
        candidate = Path(raw)
        return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate

    def secret(self, env_key_path: str) -> Optional[str]:
        """按配置里写的环境变量名去取密钥，密钥本身永不落配置文件。"""
        env_name = self.get(env_key_path)
        return os.environ.get(env_name) if env_name else None

    def as_dict(self) -> Dict[str, Any]:
        return self._data


_MISSING = object()


@dataclass(frozen=True)
class RelationSpec:
    name: str
    label: str
    domain: List[str]
    range: List[str]
    temporal: bool = False
    symmetric: bool = False


@dataclass(frozen=True)
class EntitySpec:
    name: str
    label: str
    key_attributes: List[str] = field(default_factory=list)


class SchemaRegistry:
    """本体注册表。抽取结果的合法性只在这里判定，别处不许再写类型常量。"""

    def __init__(self, entities: Dict[str, EntitySpec], relations: Dict[str, RelationSpec]) -> None:
        self._entities = entities
        self._relations = relations
        self._label_to_entity = {spec.label: name for name, spec in entities.items()}
        self._label_to_relation = {spec.label: name for name, spec in relations.items()}

    @property
    def entity_types(self) -> List[str]:
        return list(self._entities)

    @property
    def relation_types(self) -> List[str]:
        return list(self._relations)

    def entity_spec(self, name: str) -> Optional[EntitySpec]:
        return self._entities.get(name)

    def relation_spec(self, name: str) -> Optional[RelationSpec]:
        return self._relations.get(name)

    def canonical_entity_type(self, raw: str) -> Optional[str]:
        """把模型可能吐出的中文标签或大小写变体折回标准类型名。"""
        token = (raw or "").strip()
        if token in self._entities:
            return token
        if token in self._label_to_entity:
            return self._label_to_entity[token]
        lowered = token.lower()
        for name in self._entities:
            if name.lower() == lowered:
                return name
        return None

    def canonical_relation_type(self, raw: str) -> Optional[str]:
        token = (raw or "").strip()
        if token in self._relations:
            return token
        if token in self._label_to_relation:
            return self._label_to_relation[token]
        lowered = token.lower().replace(" ", "_").replace("-", "_")
        for name in self._relations:
            if name.lower() == lowered:
                return name
        return None

    def validate_triple(self, head_type: str, relation: str, tail_type: str) -> Optional[str]:
        """合法返回 None，不合法返回可写进日志的中文原因。"""
        spec = self._relations.get(relation)
        if spec is None:
            return f"未知关系类型 {relation!r}"
        if head_type not in self._entities:
            return f"未知头实体类型 {head_type!r}"
        if tail_type not in self._entities:
            return f"未知尾实体类型 {tail_type!r}"
        if head_type not in spec.domain:
            return f"关系 {relation} 的头实体类型应为 {spec.domain}，实际为 {head_type}"
        if tail_type not in spec.range:
            return f"关系 {relation} 的尾实体类型应为 {spec.range}，实际为 {tail_type}"
        return None

    def describe(self) -> Dict[str, Any]:
        return {
            "entity_types": {n: s.label for n, s in self._entities.items()},
            "relation_types": {
                n: {"label": s.label, "domain": s.domain, "range": s.range}
                for n, s in self._relations.items()
            },
        }

    @classmethod
    def from_file(cls, path: Path) -> "SchemaRegistry":
        if not path.exists():
            raise ConfigError(f"本体文件不存在: {path}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        raw_entities = payload.get("entity_types") or {}
        raw_relations = payload.get("relation_types") or {}
        if not raw_entities or not raw_relations:
            raise ConfigError("本体文件必须同时定义 entity_types 与 relation_types")

        entities = {
            name: EntitySpec(
                name=name,
                label=str(body.get("label", name)),
                key_attributes=list(body.get("key_attributes", [])),
            )
            for name, body in raw_entities.items()
        }

        relations: Dict[str, RelationSpec] = {}
        for name, body in raw_relations.items():
            domain = list(body.get("domain", []))
            tail = list(body.get("range", []))
            if not domain or not tail:
                raise ConfigError(f"关系 {name} 必须声明 domain 与 range")
            unknown = [t for t in domain + tail if t not in entities]
            if unknown:
                raise ConfigError(f"关系 {name} 引用了未定义的实体类型: {unknown}")
            relations[name] = RelationSpec(
                name=name,
                label=str(body.get("label", name)),
                domain=domain,
                range=tail,
                temporal=bool(body.get("temporal", False)),
                symmetric=bool(body.get("symmetric", False)),
            )
        return cls(entities, relations)


@lru_cache(maxsize=4)
def load_settings(path: Optional[str] = None) -> Settings:
    target = Path(path) if path else DEFAULT_SETTINGS_FILE
    if not target.exists():
        raise ConfigError(f"配置文件不存在: {target}")
    data = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    return Settings(data, target)


@lru_cache(maxsize=4)
def load_schema(path: Optional[str] = None) -> SchemaRegistry:
    target = Path(path) if path else load_settings().path("paths.schema_file")
    return SchemaRegistry.from_file(Path(target))


def read_prompt(template_name: str) -> str:
    """从 config/prompts 读取 prompt 模板，模板内容不进代码。"""
    base = load_settings().path("paths.prompt_dir")
    target = base / template_name
    if not target.exists():
        raise ConfigError(f"prompt 模板不存在: {target}")
    return target.read_text(encoding="utf-8")
