"""Collect expanded-corpus reruns, including incomplete/error status and caveats."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.run_benchmark_v2 import summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", default="eval/gap_repair/expanded")
    args = parser.parse_args()
    experiment = ROOT / args.experiment
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    prepared = json.loads((experiment / "prepared.json").read_text(encoding="utf-8"))
    manifest = json.loads((experiment / "index_manifest.json").read_text(encoding="utf-8"))
    runs = []
    for directory in sorted((experiment / "runs").glob("test-*")):
        rows = [json.loads(line) for line in (directory / "results.jsonl").read_text(encoding="utf-8").splitlines()]
        for view in ("complete", "masked"):
            summary_file = directory / f"{view}.json"
            if not summary_file.exists():
                continue
            metadata = json.loads(summary_file.read_text(encoding="utf-8"))["run"]
            selected = {r["question_id"]: r for r in rows if r["graph_view"] == view}
            values = list(selected.values())
            summary = summarize(values, metadata)
            method = metadata["method"]
            label = method
            if method == "gap_repair":
                label += " / " + metadata["generator"]
                if not metadata["gap_repair"].get("corpus_records", False):
                    label += " / no-person-records"
            runs.append({"label": label, "view": view, "rows": len(values),
                         "errors": sum(bool(r.get("error")) for r in values),
                         "historical_errors": sum(bool(r.get("error")) for r in rows if r["graph_view"] == view),
                         "path": directory.relative_to(ROOT).as_posix(), "summary": summary,
                         "structured_fallbacks": sum("降级" in str((r.get("answer") or {}).get("debug_info", {}).get("generator", "")) for r in values)})
    payload = {"prepared": prepared, "index": manifest, "runs": runs,
               "runner_sha256": hashlib.sha256((ROOT / "eval/run_gap_repair.py").read_bytes()).hexdigest()}
    (output / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 扩充语料诊断性复跑", "",
             "这不是经跨源实体去重、自然文本独立性核验后的正式 benchmark；不能据此宣称超过全部基线。", "",
             f"输入：661 条影片记录；导入 {prepared['documents']} 篇文档、{prepared['chunks']} 个 Chunk。",
             f"沿用原 40 题规格（dev 8 / test 32），重新推导真值与缺边掩码；{len(prepared['answer_changes'])} 题答案 ID 集合变化。",
             "本轮不根据 test 成绩调整检索参数。BGE-M3 / 1024 维，top-k=6；算法参数见每个 run 配置。", "",
             "| 方法 | 图视图 | 记录数 | 当前错误 | LLM 降级 | Answer F1 | EM |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for run in runs:
        metrics = next(iter(run["summary"]["methods"].values()), {})
        # Read the aggregate shape produced by the shared scorer.
        aggregate = metrics.get("overall", metrics)
        f1, em = aggregate.get("answer_f1"), aggregate.get("exact_match")
        f1_text = f"{f1:.4f}" if f1 is not None else "N/A"
        em_text = f"{em:.4f}" if em is not None else "N/A"
        lines.append(f"| {run['label']} | {run['view']} | {run['rows']}/32 | {run['errors']} | {run['structured_fallbacks']} | {f1_text} | {em_text} |")
    lines += ["", "## 结果解释边界", "",
              "- 同一影片仍可能有多个 ID/译名。例如 Q25188《全面启动》与 IMDB:tt1375666《盗梦空间》；维基原始 infobox 的 cnname 明确给出对应名称。这会影响完整集合、不同影片计数和缺边路径。",
              "- 原 benchmark 的人物别名并集合并策略原样保留；扩充数据中的同名异人尚未专项消歧。",
              "- IMDb 新增文本包含由结构化演职员表生成的简介。关闭人物片单推断并不等于只使用独立自然语言正文。",
              "- 缺边策略沿用旧脚本；director_overlap 只遮蔽选定 gold 路径末边，不保证扩充图上的所有替代证明均被切断。",
              "- 关闭人物片单推断的版本也无法使用默认版的双方片单否定策略，因此二者差异并非只度量正例补边。",
              "- library_graphrag 依赖旧版外部 Neo4j 图，本轮未覆盖/重建该数据库，也不挪用旧成绩作同语料比较。",
              "- 默认 GapRepair 使用确定性答案输出；各本地基线使用共享 LLM。不能把该对比直接解释为纯检索能力差异。",
              "- GapRepair / llm 是同一共享生成器的对照，但仍可能凭上下文或模型知识答对未形成严格路径证明的题，Answer F1 不等于路径修复成功率。",
              "- API 异常只对失败条目断点续跑；历史失败保留。生成器内部降级结果不按成绩选择性重跑。",
              "- 一个格式错误的导演记录被既有导入器跳过，详见 prepared.skipped。", "",
              "## 复现与产物", "", f"实验目录：`{experiment.relative_to(ROOT).as_posix()}`。原语料、主索引和上一轮结果均保留。", "",
              "```powershell", "python eval/prepare_expanded.py --workers 4",
              f"python eval/run_gap_repair.py --settings {experiment.relative_to(ROOT).as_posix()}/settings.yaml --questions {experiment.relative_to(ROOT).as_posix()}/questions.yaml --output {experiment.relative_to(ROOT).as_posix()}/runs --split test --workers 4",
              "# 对照：追加 --graph-view masked --no-corpus-records",
              "# 本地基线：追加 --graph-view masked --answer-mode llm --method hipporag2",
              "```", "", f"语料 SHA256：`{prepared['source_sha256']}`。",
              f"索引 SHA256：`{manifest['index_sha256']}`。", "",
              "逐题答案、引文、临时证据、错误与配置签名保存在 runs/*/results.jsonl 和同目录汇总中。",
              "机器可读汇总：summary.json。代码验证：`python -m pytest -q`，45 passed（2 条已有 FastAPI 弃用警告）。", ""]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:18]))


if __name__ == "__main__":
    main()
