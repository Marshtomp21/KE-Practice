#!/usr/bin/env python3
"""Generate GraphRAG community reports through DeepSeek JSON-object mode.

Microsoft GraphRAG v3.1.2 asks its provider for a Pydantic JSON Schema in the
community-report workflow.  DeepSeek V4 Flash currently accepts JSON-object
mode but not that JSON-Schema form.  This adapter preserves GraphRAG's graph,
community and report data contract while replacing only that API call.

It deliberately lives outside ``upstream/`` so the official checkout remains
unchanged.  Outputs are marked as an API-adapted reproduction in the docs.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = ROOT / "workspace_l2_v2"
DEFAULT_SOURCE_ENV = ROOT.parent / "kg2rag" / "config" / "api.env"


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("\"").strip("'")
    return values


def required(values: dict[str, str], key: str) -> str:
    value = values.get(key, "").strip()
    if not value:
        raise SystemExit(f"Missing {key} in {DEFAULT_SOURCE_ENV}.")
    return value


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value) if isinstance(value, (list, tuple, set)) else [value]


def build_context(community: dict[str, Any], entities: pd.DataFrame, relationships: pd.DataFrame) -> str:
    entity_ids = {str(item) for item in as_list(community["entity_ids"])}
    relationship_ids = {str(item) for item in as_list(community["relationship_ids"])}
    entity_rows = entities[entities["id"].astype(str).isin(entity_ids)]
    relation_rows = relationships[relationships["id"].astype(str).isin(relationship_ids)]

    entity_lines = ["id,title,description"]
    for row in entity_rows.itertuples(index=False):
        entity_lines.append(f"{row.human_readable_id},{row.title},{row.description}")
    relation_lines = ["id,source,target,description"]
    for row in relation_rows.itertuples(index=False):
        relation_lines.append(
            f"{row.human_readable_id},{row.source},{row.target},{row.description}"
        )
    return "Entities\n" + "\n".join(entity_lines) + "\n\nRelationships\n" + "\n".join(relation_lines)


def report_prompt(context: str) -> str:
    return f"""You are preparing a concise report from a knowledge-graph community.
Use only the supplied records; do not invent facts. Return a JSON object with
exactly these fields: title (string), summary (string), rating (number 0-10),
rating_explanation (string), findings (array of 2-5 objects, each with summary
and explanation strings). Ground statements with [Data: Entities (id); Relationships (id)].
Keep the total report under 500 words.

{context}
"""


def call_deepseek(values: dict[str, str], prompt: str, insecure_local_proxy: bool) -> dict[str, Any]:
    endpoint = values.get("REPRO_LLM_ENDPOINT", "https://api.deepseek.com/chat/completions")
    base_url = endpoint.rsplit("/chat/completions", 1)[0]
    # Some campus/local network gateways install a self-signed TLS certificate.
    # Opting into this mode is explicit and scoped to this one reproduction call.
    http_client = httpx.Client(verify=False) if insecure_local_proxy else None
    client = OpenAI(api_key=required(values, "REPRO_LLM_API_KEY"), base_url=base_url, http_client=http_client)
    response = client.chat.completions.create(
        model=values.get("REPRO_LLM_MODEL", "deepseek-v4-flash"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1200,
        response_format={"type": "json_object"},
        extra_body={"thinking": {"type": "disabled"}},
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("DeepSeek returned an empty response.")
    try:
        return json.loads(content)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"DeepSeek did not return valid JSON: {content[:300]}") from error


def validate_report(raw: dict[str, Any]) -> dict[str, Any]:
    findings = raw.get("findings")
    if not isinstance(findings, list) or not findings:
        raise ValueError("Report has no findings list.")
    normalized_findings = []
    for finding in findings:
        if not isinstance(finding, dict) or not finding.get("summary") or not finding.get("explanation"):
            raise ValueError("Each finding needs summary and explanation.")
        normalized_findings.append({"summary": str(finding["summary"]), "explanation": str(finding["explanation"])})
    rating = float(raw["rating"])
    if not 0 <= rating <= 10:
        raise ValueError("rating must be between 0 and 10.")
    return {
        "title": str(raw["title"]), "summary": str(raw["summary"]), "rating": rating,
        "rating_explanation": str(raw["rating_explanation"]), "findings": normalized_findings,
    }


def full_content(report: dict[str, Any]) -> str:
    sections = "\n\n".join(f"## {f['summary']}\n\n{f['explanation']}" for f in report["findings"])
    return f"# {report['title']}\n\n{report['summary']}\n\n{sections}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--source-env", type=Path, default=DEFAULT_SOURCE_ENV)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-api", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--insecure-local-proxy", action="store_true", help="Allow the local development proxy's self-signed TLS certificate.")
    args = parser.parse_args()
    output_dir = args.workspace / "output"
    communities_path = output_dir / "communities.parquet"
    print(json.dumps({"method": "Microsoft GraphRAG community-report DeepSeek adapter", "workspace": str(args.workspace), "output": str(output_dir), "uses": "DeepSeek JSON-object mode"}, ensure_ascii=False))
    if not communities_path.exists():
        raise SystemExit(f"Missing partial official index output: {communities_path}")
    communities = pd.read_parquet(communities_path)
    if args.dry_run:
        print(json.dumps({"communities": len(communities), "api_calls": len(communities), "writes": "community_reports.parquet"}))
        return
    if not args.allow_api:
        raise SystemExit("Refusing remote calls; inspect --dry-run then pass --allow-api.")
    if not args.source_env.exists():
        raise SystemExit(f"Credential source not found: {args.source_env}")
    values = load_env(args.source_env)
    entities = pd.read_parquet(output_dir / "entities.parquet")
    relationships = pd.read_parquet(output_dir / "relationships.parquet")

    partial_path = output_dir / "community_reports.partial.jsonl"
    completed: dict[int, dict[str, Any]] = {}
    if partial_path.exists():
        for line in partial_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed[int(row["community"])] = row
        print(json.dumps({"checkpoint_reports": len(completed), "path": str(partial_path)}, ensure_ascii=False))
    audit: list[dict[str, Any]] = []
    for community in communities.to_dict(orient="records"):
        community_id = int(community["community"])
        if community_id in completed:
            continue
        context = build_context(community, entities, relationships)
        last_error: Exception | None = None
        for attempt in range(1, args.retries + 1):
            try:
                report = validate_report(call_deepseek(values, report_prompt(context), args.insecure_local_proxy))
                break
            except Exception as error:
                last_error = error
                if attempt == args.retries:
                    raise RuntimeError(f"community {community_id} failed after {args.retries} attempts") from error
                time.sleep(2 ** (attempt - 1))
        row = {
            "id": f"deepseek-adapter-community-{community_id}",
            "human_readable_id": community_id, "community": community_id,
            "level": int(community["level"]), "parent": int(community["parent"]),
            "children": as_list(community["children"]), "title": report["title"],
            "summary": report["summary"], "full_content": full_content(report), "rank": report["rating"],
            "rating_explanation": report["rating_explanation"], "findings": report["findings"],
            "full_content_json": json.dumps(report, ensure_ascii=False, indent=2),
            "period": str(community["period"]), "size": int(community["size"]),
        }
        completed[community_id] = row
        with partial_path.open("a", encoding="utf-8") as checkpoint:
            checkpoint.write(json.dumps(row, ensure_ascii=False, default=lambda value: value.item() if hasattr(value, "item") else str(value)) + "\n")
        audit.append({"community": community_id, "entities": len(as_list(community["entity_ids"])), "relationships": len(as_list(community["relationship_ids"])), "report_title": report["title"]})
        print(f"[report {len(completed):03d}/{len(communities)}] community {community_id}", flush=True)
    missing = set(int(value) for value in communities["community"]) - set(completed)
    if missing:
        raise RuntimeError(f"Missing community reports: {sorted(missing)}")
    rows = [completed[int(community["community"])] for community in communities.to_dict(orient="records")]
    reports = pd.DataFrame(rows)
    reports.to_parquet(output_dir / "community_reports.parquet", index=False)
    (output_dir / "deepseek_adapter_audit.json").write_text(json.dumps({"created_at": datetime.now(timezone.utc).isoformat(), "adapter": "DeepSeek JSON-object -> GraphRAG community_reports schema", "community_reports": len(reports), "new_reports_this_invocation": len(audit), "communities": audit}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed", "community_reports": len(reports), "path": str(output_dir / "community_reports.parquet")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
