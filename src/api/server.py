"""FastAPI 后端。

只做三件事：把问题转交 QAService、把子图整理成前端好画的形状、把静态页面挂上。
资源（图、索引）在启动时加载一次，后续请求不再读盘。
任何异常都转成带 detail 的 JSON，前端据此显示提示而不是白屏。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..core.config import load_schema, load_settings
from ..generate.service import QAService
from ..methods.library_graphrag import MethodUnavailable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = PROJECT_ROOT / "web"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    retriever: Optional[str] = None
    top_k: Optional[int] = Field(default=None, ge=1, le=50)


def _graph_payload(answer) -> Dict[str, Any]:
    """把子图转成 { nodes, edges } —— 前端画图库直接吃这个形状。"""
    schema = load_schema()
    scores = answer.subgraph.node_scores
    highlight = set(answer.subgraph.highlight_path)
    nodes = [
        {
            "id": entity.id,
            "label": entity.name,
            "type": entity.type,
            "type_label": (schema.entity_spec(entity.type).label if schema.entity_spec(entity.type) else entity.type),
            "score": round(float(scores.get(entity.id, 0.0)), 4),
            "highlight": entity.id in highlight,
            "aliases": entity.aliases,
            "evidences": [e.to_dict() for e in entity.evidences[:5]],
        }
        for entity in answer.subgraph.entities
    ]
    edges = [
        {
            "id": relation.id,
            "source": relation.head_id,
            "target": relation.tail_id,
            "type": relation.type,
            "label": (schema.relation_spec(relation.type).label if schema.relation_spec(relation.type) else relation.type),
            "start_year": relation.start_year,
            "end_year": relation.end_year,
            "highlight": relation.head_id in highlight and relation.tail_id in highlight,
            "evidences": [e.to_dict() for e in relation.evidences[:3]],
        }
        for relation in answer.subgraph.relations
    ]
    for evidence in answer.subgraph.evidence_nodes:
        chunk = evidence.chunk
        confidences = [float(span.get("confidence", 1.0))
                       for relation in evidence.supported_relations
                       for span in relation.get("evidences", [])]
        weak_record = any(relation.get("attributes", {}).get("evidence_tier") == "dataset_assertion"
                          for relation in evidence.supported_relations)
        nodes.append({
            "id": evidence.id, "label": "临时证据：" + str(chunk.metadata.get("title", chunk.doc_id)),
            "type": "EvidenceNode", "type_label": "片单推断证据" if weak_record else "本次问答证据", "temporary": True,
            "score": 1.0, "highlight": True, "aliases": [],
            "evidences": [{"doc_id": chunk.doc_id, "chunk_id": chunk.id,
                           "char_start": chunk.char_offset, "char_end": chunk.char_end,
                           "raw_text": chunk.text, "confidence": min(confidences, default=0.0)}],
        })
        for entity_id in evidence.entity_ids:
            edges.append({"id": evidence.id + "|" + entity_id,
                          "source": evidence.id, "target": entity_id,
                          "type": "supports", "label": "文本支持（临时）",
                          "temporary": True, "highlight": True,
                          "evidences": nodes[-1]["evidences"]})
    return {"nodes": nodes, "edges": edges}


def create_app() -> FastAPI:
    settings = load_settings()
    app = FastAPI(title="影视 GraphRAG 问答", version="1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.get("api.cors_origins", ["*"])),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    state: Dict[str, Any] = {"service": None, "error": None}

    def service() -> QAService:
        if state["service"] is None:
            try:
                state["service"] = QAService(settings)
            except FileNotFoundError as exc:
                state["error"] = str(exc)
                raise HTTPException(
                    status_code=503,
                    detail=f"索引尚未构建：{exc}。请先运行 python scripts/build_index.py",
                ) from exc
        return state["service"]

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        try:
            active = service()
        except HTTPException as exc:
            return {"ready": False, "detail": exc.detail}
        return {
            "ready": True,
            "retrievers": active.retriever_names,
            "default_retriever": active.default_retriever,
            "graph": active.graph_stats(),
        }

    @app.get("/api/schema")
    def schema() -> Dict[str, Any]:
        return load_schema().describe()

    @app.get("/api/examples")
    def examples(limit: int = 6) -> Dict[str, Any]:
        """从 Benchmark v2 开发集选择示例；不应用评测掩码或暴露 gold。"""
        if limit <= 0:
            return {"examples": []}
        question_file = PROJECT_ROOT / "eval" / "benchmark_v2" / "questions.yaml"
        if not question_file.exists():
            return {"examples": []}
        try:
            payload = yaml.safe_load(question_file.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            return {"examples": []}

        labels = {
            "cofilm_set": "共同出演",
            "director_overlap": "共同演员",
            "repeated_cast": "重复合作",
            "multi_director": "多导演影片",
            "director_actor_films": "导演与演员交集",
            "hard_negative": "困难负例",
        }
        picked: List[Dict[str, str]] = []
        seen_kinds: set = set()
        # 开发集每种题型选一道；冻结测试集不用于日常调试
        for question in payload.get("questions", []):
            if question.get("split") != "dev":
                continue
            kind = question.get("kind", "")
            if kind in seen_kinds:
                continue
            seen_kinds.add(kind)
            picked.append(
                {
                    "kind": kind,
                    "label": labels.get(kind, kind),
                    "question": question.get("question", ""),
                }
            )
            if len(picked) >= limit:
                break
        return {"examples": picked}

    @app.post("/api/ask")
    def ask(request: AskRequest) -> Dict[str, Any]:
        active = service()
        if request.retriever and request.retriever not in active.retriever_names:
            raise HTTPException(
                status_code=400,
                detail=f"未知检索器 {request.retriever}，可选：{active.retriever_names}",
            )
        try:
            answer = active.ask(
                request.question, retriever_name=request.retriever, top_k=request.top_k
            )
        except MethodUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"问答过程出错：{exc}") from exc

        return {
            "question": request.question,
            "retriever": answer.retriever_name,
            "latency": round(answer.latency, 3),
            "answer": answer.text,
            "citations": [c.to_dict() for c in answer.citations],
            "graph": _graph_payload(answer),
            "debug": answer.debug_info,
        }

    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(str(WEB_DIR / "index.html"))

    @app.on_event("shutdown")
    def shutdown() -> None:
        if state["service"] is not None:
            state["service"].close()

    return app


app = create_app()
