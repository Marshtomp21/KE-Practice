#!/usr/bin/env python3
"""Re-score a saved result file with the frozen benchmark's auditable rules."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_utils import read_jsonl, score_answer, summarize_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="JSON summary containing a results array")
    parser.add_argument("--output", type=Path, default=None, help="Defaults to <input>.rescored.json")
    args = parser.parse_args()
    questions = {row["id"]: row for row in read_jsonl(args.questions)}
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    results = payload.get("results")
    if not isinstance(results, list):
        raise SystemExit("Input JSON must contain a results array.")
    for row in results:
        question = questions.get(row.get("id"))
        if question is None:
            raise SystemExit(f"Unknown or out-of-split question id: {row.get('id')}")
        metrics = score_answer(question, str(row.get("answer", "")))
        row["metrics"] = metrics
        row["score"] = metrics["score"]
    payload["evaluation"] = summarize_results(results)
    payload["scorer"] = "shared/scripts/benchmark_utils.py:v2"
    latencies = [float(row["query_seconds"]) for row in results if row.get("query_seconds") is not None]
    if latencies:
        payload["mean_query_seconds"] = round(sum(latencies) / len(latencies), 3)
    output = args.output or args.input.with_name(args.input.stem + ".rescored.json")
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "evaluation": payload["evaluation"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
