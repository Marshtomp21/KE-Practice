"""汇总 complete/masked 配对结果，量化缺边退化与补偿恢复。"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = PROJECT_ROOT / "eval" / "results" / "benchmark_v2"
DEFAULT_QUESTIONS = PROJECT_ROOT / "eval" / "benchmark_v2" / "questions.yaml"


def nested(row: dict, path: str) -> Any:
    value: Any = row
    for key in path.split("."):
        value = value.get(key) if isinstance(value, dict) else None
    return value


def average(rows: Iterable[dict], path: str) -> float | None:
    values = [nested(row, path) for row in rows]
    numbers = [float(value) for value in values if isinstance(value, (int, float))]
    return statistics.mean(numbers) if numbers else None


def load_latest(path: Path) -> tuple[dict[tuple[str, str, str], dict], list[str]]:
    latest: dict[tuple[str, str, str], dict] = {}
    signature_order: list[str] = []
    if not path.exists():
        return latest, signature_order
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("error"):
            signature = str(row.get("run_signature") or "legacy-unversioned")
            latest[(row["question_id"], row["method"], signature)] = row
            if signature not in signature_order:
                signature_order.append(signature)
    return latest, signature_order


def fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--output", default="")
    parser.add_argument("--run-signature", default="")
    args = parser.parse_args()

    root = Path(args.result_dir)
    complete, complete_order = load_latest(root / "complete" / "results.jsonl")
    masked, masked_order = load_latest(root / "masked" / "results.jsonl")
    benchmark = yaml.safe_load(Path(args.questions).read_text(encoding="utf-8"))
    gap_ids = {
        item["id"] for item in benchmark["questions"]
        if item.get("graph_perturbation", {}).get("expected_gap")
    }
    complete_signatures = {signature for _, _, signature in complete}
    masked_signatures = {signature for _, _, signature in masked}
    common_signatures = complete_signatures & masked_signatures
    if args.run_signature:
        signature = args.run_signature
        if signature not in common_signatures:
            raise SystemExit("指定运行签名没有同时具备 complete 与 masked 结果")
    else:
        candidates = [item for item in masked_order if item in common_signatures]
        if not candidates:
            candidates = [item for item in complete_order if item in common_signatures]
        if not candidates:
            raise SystemExit("complete/masked 没有相同运行签名，拒绝混合不同模型结果")
        signature = candidates[-1]
    methods = sorted(
        {method for _, method, item in complete if item == signature}
        | {method for _, method, item in masked if item == signature}
    )
    if not methods:
        raise SystemExit("尚无 complete/masked 实验结果可汇总")

    lines = [
        "# Benchmark v2 配对结果", "", f"运行签名：`{signature}`", "",
        "以下指标只统计 30 道可恢复缺边题；正数 `F1 Drop` 表示缺边造成性能下降。", "",
        "| 方法 | 配对题数 | Complete F1 | Masked F1 | F1 Drop | Gap Acc | Comp Doc R | Recovered Path |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        ids = sorted(
            question_id for question_id in gap_ids
            if (question_id, method, signature) in complete
            and (question_id, method, signature) in masked
        )
        complete_rows = [complete[(question_id, method, signature)] for question_id in ids]
        masked_rows = [masked[(question_id, method, signature)] for question_id in ids]
        complete_f1 = average(complete_rows, "metrics.answer.f1")
        masked_f1 = average(masked_rows, "metrics.answer.f1")
        drop = (
            complete_f1 - masked_f1
            if complete_f1 is not None and masked_f1 is not None else None
        )
        lines.append(
            f"| {method} | {len(ids)} | {fmt(complete_f1)} | {fmt(masked_f1)} | "
            f"{fmt(drop)} | {fmt(average(masked_rows, 'metrics.gap.gap_detection_correct'))} | "
            f"{fmt(average(masked_rows, 'metrics.gap.compensation_document_recall'))} | "
            f"{fmt(average(masked_rows, 'metrics.retrieval.recovered_path_complete_rate'))} |"
        )

    output = Path(args.output) if args.output else root / "paired_report.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"配对汇总 -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
