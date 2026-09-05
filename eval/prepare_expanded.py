"""Snapshot an expanded corpus and build isolated, resumable evaluation assets.

No extraction, Neo4j writes, or changes to the original corpus/index. Gold is
constructed by the existing fixed query specifications, never by a QA method.
Embedding checkpoints are keyed by exact input text AND encoder settings.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_benchmark_v2 import (Dataset, DEV_IDS, NoAliasDumper,
    build_questions, iter_jsonl, source_digest, validate)
from scripts.import_wikipedia_films import build_document, build_person_document
from src.core.config import Settings, load_settings
from src.ingest.pipeline import IngestPipeline, load_persisted_chunks
from src.retrieve.embedding import HashingEmbedder, build_embedder
from src.retrieve.vector_index import ChunkVectorIndex


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def batch_key(texts, encoder):
    return hashlib.sha256(json.dumps([encoder, texts], ensure_ascii=False,
                                   sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="data/source/wikipedia_300_films_final")
    parser.add_argument("--output", default="eval/results/gap_repair_expanded")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    source = (ROOT / args.source_dir).resolve()
    digest = source_digest(source)
    target = (ROOT / args.output / digest[:12]).resolve()
    # Refuse broad or overlapping targets; never overwrite a source corpus.
    if target == ROOT or target == source or target in source.parents or source in target.parents:
        raise ValueError("Output must be an independent experiment directory")
    target.mkdir(parents=True, exist_ok=True)
    snapshot = target / "source"
    snapshot.mkdir(exist_ok=True)
    for name in ("films.jsonl", "actors.jsonl", "directors.jsonl", "relations.jsonl", "stats.json"):
        destination = snapshot / name
        if destination.exists():
            if destination.read_bytes() != (source / name).read_bytes():
                raise ValueError(f"Existing snapshot differs: {destination}")
        else:
            shutil.copy2(source / name, destination)
    if source_digest(snapshot) != digest:
        raise ValueError("Source changed during snapshot")

    config = copy.deepcopy(load_settings().as_dict())
    paths = {"dataset_dir": snapshot, "raw_dir": target / "raw",
             "interim_dir": target / "interim", "processed_dir": target / "processed",
             "chunk_file": target / "interim/chunks.jsonl",
             "embedding_file": target / "processed/chunk_vectors.npz",
             "gazetteer": target / "interim/wiki_gazetteer.txt"}
    config["paths"].update({key: str(path) for key, path in paths.items()})
    settings_path = target / "settings.yaml"
    serialized = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
    if settings_path.exists() and settings_path.read_text(encoding="utf-8") != serialized:
        raise ValueError("Configuration changed; choose a new --output directory")
    settings_path.write_text(serialized, encoding="utf-8")
    settings = Settings(config, settings_path)
    ready = target / "prepared.json"
    if not ready.exists():
        dataset = Dataset(snapshot)
        questions = build_questions(dataset)
        for question in questions:
            question["split"] = "dev" if question["id"] in DEV_IDS else "test"
        validate(questions)
        benchmark = {"benchmark": {"id": "film-graph-rag-expanded-40",
                     "version": 2, "status": "frozen", "question_count": len(questions),
                     "split_counts": dict(Counter(q["split"] for q in questions)),
                     "source_sha256": digest,
                     "construction": "Original fixed 40 query specs; gold and masks rederived on expanded source"},
                     "entity_catalog": {key: dataset.item(key) for key in sorted(dataset.entities)},
                     "questions": questions}
        (target / "questions.yaml").write_text(yaml.dump(benchmark, Dumper=NoAliasDumper,
            allow_unicode=True, sort_keys=False), encoding="utf-8")
        paths["raw_dir"].mkdir(exist_ok=True)
        imported, skipped = Counter(), []
        for filename, role in (("films.jsonl", "Movie"), ("actors.jsonl", "Actor"), ("directors.jsonl", "Director")):
            for row in iter_jsonl(snapshot / filename):
                document = build_document(row) if role == "Movie" else build_person_document(row, role)
                if document is None:
                    skipped.append({"role": role, "id": (row.get("film") or row.get("person") or {}).get("id")})
                    continue
                file_name = document.pop("file_name")
                dump(paths["raw_dir"] / file_name, document)
                imported[role] += 1
        pipeline = IngestPipeline(settings)
        documents, chunks, report = pipeline.run()
        if report.skipped:
            raise ValueError(f"Ingest errors: {report.skipped[:5]}")
        pipeline.persist(documents, chunks)
        names = sorted({name for item in dataset.entities.values()
                        for name in [item["name"], *item.get("aliases", [])]})
        paths["gazetteer"].write_text("\n".join(names) + "\n", encoding="utf-8")
        old = yaml.safe_load((ROOT / "eval/benchmark_v2/questions.yaml").read_text(encoding="utf-8"))
        old_questions = {q["id"]: q for q in old["questions"]}
        changes = []
        for question in questions:
            before = {a["id"] for a in old_questions[question["id"]]["gold_answers"]}
            after = {a["id"] for a in question["gold_answers"]}
            if before != after:
                changes.append({"id": question["id"], "before": sorted(before), "after": sorted(after)})
        dump(ready, {"source_sha256": digest, "imported": dict(imported), "skipped": skipped,
                     "documents": len(documents), "chunks": len(chunks), "answer_changes": changes,
                     "gold_canonicalization": "Original benchmark alias union policy retained; not identity-disambiguated"})
    print(ready.read_text(encoding="utf-8"), flush=True)
    print("Assets:", target, flush=True)
    if args.prepare_only:
        return
    if paths["embedding_file"].exists():
        print("Existing isolated index retained", flush=True)
        return
    chunks = load_persisted_chunks(settings)
    encoder = {key: settings.get(f"embedding.{key}") for key in
               ("provider", "endpoint", "model", "dimension")}
    if isinstance(build_embedder(settings), HashingEmbedder):
        raise ValueError("Remote embedding credentials required; no silent hashing fallback")
    cache = target / "embedding_batches"
    cache.mkdir(exist_ok=True)
    batches = [chunks[i:i + 32] for i in range(0, len(chunks), 32)]

    def encode(batch):
        texts = [f"{c.metadata.get('title', '')}\n{c.text}" for c in batch]
        path = cache / (batch_key(texts, encoder) + ".npz")
        if path.exists():
            with np.load(path) as saved:
                matrix = saved["matrix"]
        else:
            for attempt in range(3):
                try:
                    matrix = build_embedder(settings).encode(texts)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
            tmp = path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, matrix=matrix)
            tmp.replace(path)
        if matrix.shape != (len(batch), encoder["dimension"]) or not np.isfinite(matrix).all():
            raise ValueError(f"Invalid embedding checkpoint: {path}")
        return matrix

    matrices = [None] * len(batches)
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 8))) as executor:
        futures = {executor.submit(encode, batch): i for i, batch in enumerate(batches)}
        for completed, future in enumerate(as_completed(futures), 1):
            matrices[futures[future]] = future.result()
            if completed % 10 == 0 or completed == len(batches):
                print(f"Embedding batches: {completed}/{len(batches)}", flush=True)
    index = ChunkVectorIndex(settings)
    index._chunks = chunks
    index._matrix = np.concatenate(matrices)
    index.persist()
    dump(target / "index_manifest.json", {"encoder": encoder, "source_sha256": digest,
         "chunks": len(chunks), "dimension": index.dimension,
         "index_sha256": hashlib.sha256(paths["embedding_file"].read_bytes()).hexdigest()})
    print("Index complete:", paths["embedding_file"], flush=True)


if __name__ == "__main__":
    main()
