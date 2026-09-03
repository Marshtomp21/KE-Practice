#!/usr/bin/env python3
"""Create an isolated Microsoft GraphRAG workspace from benchmark v2."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT.parent / "shared" / "benchmarks" / "l2_film_120_v2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--workspace", type=Path, default=ROOT / "workspace_l2_v2")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    documents = [json.loads(line) for line in (args.corpus / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    print(json.dumps({"workspace": str(args.workspace), "documents": len(documents), "source": str(args.corpus)}, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    args.workspace.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "template" / "settings.yaml", args.workspace / "settings.yaml")
    source_prompts = ROOT / "template" / "prompts"
    target_prompts = args.workspace / "prompts"
    if target_prompts.exists():
        shutil.rmtree(target_prompts)
    shutil.copytree(source_prompts, target_prompts)
    input_dir = args.workspace / "input"
    input_dir.mkdir(exist_ok=True)
    for row in documents:
        shutil.copy2(args.corpus / row["path"], input_dir / f"{row['doc_id']}.txt")


if __name__ == "__main__":
    main()
