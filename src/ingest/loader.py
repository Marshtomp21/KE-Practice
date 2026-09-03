"""从 data/raw 批量读取条目。

单条失败只记账不抛出，保证批量导入不被一个坏文件打断（F1）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from ..core.interfaces import DocumentSource
from ..core.types import RawDocument

REQUIRED_FIELDS = ("doc_id", "title", "entity_type", "text")


@dataclass
class LoadReport:
    """一次导入的账本，导入脚本直接打印它。"""

    accepted: int = 0
    skipped: List[Dict[str, str]] = field(default_factory=list)

    def skip(self, source: str, reason: str) -> None:
        self.skipped.append({"source": source, "reason": reason})

    def summary(self) -> str:
        return f"成功 {self.accepted} 条，跳过 {len(self.skipped)} 条"


class JsonDirectorySource(DocumentSource):
    """把一个目录下的 *.json 条目读成 RawDocument 流。

    already_done 用于增量导入：调用方把上一轮已处理的 doc_id 传进来即可跳过。
    """

    def __init__(
        self,
        directory: Path,
        already_done: Optional[Iterable[str]] = None,
        report: Optional[LoadReport] = None,
    ) -> None:
        self.directory = Path(directory)
        self.already_done = set(already_done or ())
        self.report = report or LoadReport()

    def iter_documents(self) -> Iterator[RawDocument]:
        if not self.directory.exists():
            self.report.skip(str(self.directory), "目录不存在")
            return
        for file_path in sorted(self.directory.glob("*.json")):
            document = self._read_one(file_path)
            if document is None:
                continue
            if document.doc_id in self.already_done:
                self.report.skip(file_path.name, "已处理，增量跳过")
                continue
            self.report.accepted += 1
            yield document

    def _read_one(self, file_path: Path) -> Optional[RawDocument]:
        try:
            payload: Dict[str, Any] = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.report.skip(file_path.name, f"读取失败: {exc}")
            return None

        missing = [f for f in REQUIRED_FIELDS if not str(payload.get(f, "")).strip()]
        if missing:
            self.report.skip(file_path.name, f"缺少字段 {missing}")
            return None

        return RawDocument(
            doc_id=str(payload["doc_id"]).strip(),
            title=str(payload["title"]).strip(),
            url=str(payload.get("url", "")).strip(),
            entity_type=str(payload["entity_type"]).strip(),
            text=str(payload["text"]),
            infobox=dict(payload.get("infobox") or {}),
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "source_kind": "json_directory",
            "directory": str(self.directory),
            "accepted": self.report.accepted,
            "skipped": len(self.report.skipped),
        }
