#!/usr/bin/env python3
"""Run the official GraphRAG indexer with the shared API configuration.

The wrapper keeps credentials out of shell history and records wall-clock time.
With DeepSeek the final community-report workflow is expected to fail because
the provider does not accept GraphRAG's JSON-schema response mode; the separate
``build_community_reports_deepseek.py`` adapter completes that one artifact.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=ROOT / "workspace_l2_v2")
    parser.add_argument("--env-file", type=Path, default=ROOT.parent / "kg2rag" / "config" / "api.env")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-api", action="store_true")
    args = parser.parse_args()
    command = [str(ROOT.parent / ".venv-microsoft-graphrag" / "bin" / "graphrag"), "index", "--root", str(args.workspace)]
    if args.resume:
        command.append("--resume")
    print(json.dumps({"workspace": str(args.workspace), "command": command, "expected_provider_limitation": "community report JSON schema"}, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    if not args.allow_api:
        raise SystemExit("Pass --allow-api after --dry-run.")
    values = load_env(args.env_file)
    process_env = os.environ.copy()
    process_env["GRAPHRAG_LLM_API_KEY"] = values["REPRO_LLM_API_KEY"]
    process_env["GRAPHRAG_EMBED_API_KEY"] = values["REPRO_EMBED_API_KEY"]
    started = time.perf_counter()
    result = subprocess.run(command, env=process_env)
    audit = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "official_index_returncode": result.returncode,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "resume": args.resume,
        "community_reports_adapter_required": result.returncode != 0,
    }
    args.workspace.mkdir(parents=True, exist_ok=True)
    (args.workspace / "official_index_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
