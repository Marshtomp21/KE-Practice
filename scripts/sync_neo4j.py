"""把本地 Chunk 向量和数据集自带关系增量同步到 Neo4j。

本脚本只负责方法运行所需的最薄适配，不参与日常问答。它使用 MERGE 增量写入，
不会清空数据库。运行前先安装 requirements-graphrag.txt 并配置 NEO4J_* 环境变量。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import load_settings
from src.retrieve.vector_index import ChunkVectorIndex


def batches(items: Sequence[dict], size: int) -> Iterable[List[dict]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def relation_id(head_id: str, relation_type: str, tail_id: str) -> str:
    token = f"{head_id}|{relation_type}|{tail_id}"
    return hashlib.sha1(token.encode("utf-8")).hexdigest()


def load_dataset_graph(
    dataset_dir: Path, index: ChunkVectorIndex
) -> Tuple[List[dict], List[dict], List[dict]]:
    """直接读取数据集自带的实体和关系，不再依赖本地抽取生成的图快照。"""
    required = ["films.jsonl", "actors.jsonl", "directors.jsonl", "relations.jsonl"]
    missing = [name for name in required if not (dataset_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"真实数据集缺少文件：{missing}；目录：{dataset_dir}")

    entities: Dict[str, dict] = {}

    def add_entity(entity_id, name, entity_type) -> None:
        entity_id = str(entity_id or "").strip()
        name = str(name or "").strip()
        if not entity_id or not name:
            return
        existing = entities.get(entity_id)
        if existing is None:
            entities[entity_id] = {"id": entity_id, "name": name, "type": entity_type}
        elif existing["type"] != "Movie" and entity_type == "Person":
            existing["type"] = "Person"

    for record in iter_jsonl(dataset_dir / "films.jsonl"):
        film = record.get("film") or {}
        add_entity(film.get("id"), film.get("name"), "Movie")
    for file_name in ("actors.jsonl", "directors.jsonl"):
        for record in iter_jsonl(dataset_dir / file_name):
            person = record.get("person") or {}
            add_entity(person.get("id"), person.get("name"), "Person")

    target_types = {
        "执导": "Person", "出演": "Person", "编剧": "Person",
        "出品": "Company", "获奖": "Award", "提名": "Award",
        "类型": "Genre", "改编自": "Work", "前作": "Movie", "续作": "Movie",
    }
    relation_rows: List[dict] = []
    for row in iter_jsonl(dataset_dir / "relations.jsonl"):
        head_id = str(row.get("source_id") or "")
        tail_id = str(row.get("target_id") or "")
        kind = str(row.get("relation") or "RELATED_TO")
        add_entity(head_id, row.get("source_name"), "Movie" if row.get("source_type") == "影片" else "Entity")
        add_entity(tail_id, row.get("target_name"), target_types.get(kind, "Entity"))
        if head_id and tail_id:
            relation_rows.append({
                "id": relation_id(head_id, kind, tail_id),
                "head_id": head_id,
                "tail_id": tail_id,
                "type": kind,
                "start_year": None,
                "end_year": None,
                "roles": list(row.get("roles") or []),
                "films": [],
                "collaboration_count": None,
                "evidence_url": str(row.get("evidence_url") or ""),
                "raw_evidence": str(row.get("raw_evidence") or ""),
            })

    collaboration_file = dataset_dir / "collaborations.jsonl"
    if collaboration_file.exists():
        for row in iter_jsonl(collaboration_file):
            left = row.get("person_a") or {}
            right = row.get("person_b") or {}
            head_id = str(left.get("id") or "")
            tail_id = str(right.get("id") or "")
            add_entity(head_id, left.get("name"), "Person")
            add_entity(tail_id, right.get("name"), "Person")
            if head_id and tail_id:
                relation_rows.append({
                    "id": relation_id(head_id, "合作", tail_id),
                    "head_id": head_id,
                    "tail_id": tail_id,
                    "type": "合作",
                    "start_year": None,
                    "end_year": None,
                    "roles": [],
                    "films": [str(item.get("name") or item.get("id") or "") for item in row.get("films") or []],
                    "collaboration_count": int(row.get("collaboration_count") or 0),
                    "evidence_url": "",
                    "raw_evidence": "",
                })

    chunks_by_doc: Dict[str, List[str]] = {}
    for chunk in index.chunks:
        chunks_by_doc.setdefault(chunk.doc_id, []).append(chunk.id)
    links = set()
    for entity_id, entity in entities.items():
        doc_ids = [f"film_{entity_id}"] if entity["type"] == "Movie" else [
            f"person_actor_{entity_id}", f"person_director_{entity_id}"
        ]
        for doc_id in doc_ids:
            for chunk_id in chunks_by_doc.get(doc_id, []):
                links.add((entity_id, chunk_id))
    evidence_links = [
        {"entity_id": entity_id, "chunk_id": chunk_id}
        for entity_id, chunk_id in sorted(links)
    ]
    return list(entities.values()), relation_rows, evidence_links


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--skip-graph", action="store_true", help="只同步 Chunk 和向量")
    parser.add_argument(
        "--recreate-index", action="store_true",
        help="删除并重建配置的向量索引（更换 embedding 模型或维度时使用）",
    )
    args = parser.parse_args()

    try:
        from neo4j import GraphDatabase
        from neo4j_graphrag.indexes import create_vector_index
    except ImportError:
        print("缺少可选依赖，请先执行：pip install -r requirements-graphrag.txt")
        return 2

    settings = load_settings()
    uri = settings.secret("library_graphrag.neo4j.uri_env")
    user = settings.secret("library_graphrag.neo4j.user_env")
    password = settings.secret("library_graphrag.neo4j.password_env")
    if not uri or not user or not password:
        print("请在 .env 或系统环境变量中配置 NEO4J_URI、NEO4J_USER、NEO4J_PASSWORD")
        return 2

    index = ChunkVectorIndex(settings=settings).load()
    vectors = index.vectors
    if len(index.chunks) != len(vectors):
        raise RuntimeError("本地片段数与向量数不一致，请重新运行 scripts/build_index.py")

    chunk_rows = [
        {
            "id": chunk.id,
            "doc_id": chunk.doc_id,
            "text": chunk.text,
            "title": str(chunk.metadata.get("title", "")),
            "embedding": vectors[pos].astype(float).tolist(),
        }
        for pos, chunk in enumerate(index.chunks)
    ]
    database = settings.get("library_graphrag.neo4j.database") or None
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            session.run("CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE").consume()
            session.run("CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE").consume()
            for group in batches(chunk_rows, args.batch_size):
                session.run(
                    """
                    UNWIND $rows AS row
                    MERGE (chunk:Chunk {id: row.id})
                    SET chunk.doc_id = row.doc_id, chunk.text = row.text,
                        chunk.title = row.title, chunk.embedding = row.embedding
                    """,
                    rows=group,
                ).consume()
        print(f"已同步 Chunk：{len(chunk_rows)}，向量维度：{index.dimension}")

        index_name = str(settings.get("library_graphrag.index_name", "text_embeddings"))
        with driver.session(database=database) as session:
            exists = session.run(
                "SHOW INDEXES YIELD name WHERE name = $name RETURN count(*) AS total",
                name=index_name,
            ).single()["total"]
            if exists and args.recreate_index:
                escaped_name = index_name.replace("`", "``")
                session.run(f"DROP INDEX `{escaped_name}` IF EXISTS").consume()
                exists = 0
                print(f"已删除旧向量索引：{index_name}")
        if not exists:
            create_vector_index(
                driver,
                index_name,
                label="Chunk",
                embedding_property="embedding",
                dimensions=index.dimension,
                similarity_fn="cosine",
                neo4j_database=database,
            )
            print(f"已创建向量索引：{index_name}")
        else:
            print(f"复用已有向量索引：{index_name}（请确认其维度为 {index.dimension}）")

        if args.skip_graph:
            return 0

        dataset_dir = settings.path("paths.dataset_dir")
        entity_rows, relation_rows, evidence_links = load_dataset_graph(dataset_dir, index)
        with driver.session(database=database) as session:
            for group in batches(entity_rows, args.batch_size):
                session.run(
                    """
                    UNWIND $rows AS row
                    MERGE (entity:Entity {id: row.id})
                    SET entity.name = row.name, entity.type = row.type
                    """,
                    rows=group,
                ).consume()
            for group in batches(evidence_links, args.batch_size):
                session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (entity:Entity {id: row.entity_id})
                    MATCH (chunk:Chunk {id: row.chunk_id})
                    MERGE (entity)-[:FROM_CHUNK]->(chunk)
                    """,
                    rows=group,
                ).consume()
            for group in batches(relation_rows, args.batch_size):
                session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (head:Entity {id: row.head_id})
                    MATCH (tail:Entity {id: row.tail_id})
                    MERGE (head)-[relation:KG_RELATION {id: row.id}]->(tail)
                    SET relation.type = row.type, relation.start_year = row.start_year,
                        relation.end_year = row.end_year, relation.roles = row.roles,
                        relation.films = row.films,
                        relation.collaboration_count = row.collaboration_count,
                        relation.evidence_url = row.evidence_url,
                        relation.raw_evidence = row.raw_evidence
                    """,
                    rows=group,
                ).consume()
        print(
            f"已同步图谱：实体 {len(entity_rows)}，关系 {len(relation_rows)}，"
            f"实体-片段证据链接 {len(evidence_links)}"
        )
        return 0
    finally:
        driver.close()


if __name__ == "__main__":
    raise SystemExit(main())
