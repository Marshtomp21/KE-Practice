"""Query-local gap repair with deterministic, cited set reasoning."""
from __future__ import annotations

import time
from dataclasses import replace

from ..core.types import Answer, Relation, RetrievalConstraints, Subgraph
from ..generate.context_builder import ContextBuilder
from ..retrieve.gap_repair import GapRepairRetriever
from ._local_graph import LocalGraphQAMethod
from .registry import register


@register("gap_repair")
class GapRepairMethod(LocalGraphQAMethod):
    def _build_retriever(self, context):
        return GapRepairRetriever(context)

    def ask(self, question: str, top_k=None, constraints: RetrievalConstraints | None = None) -> Answer:
        started = time.perf_counter()
        result = self.retriever.retrieve(question, top_k=top_k or self.default_top_k, constraints=constraints)
        if "plan" not in result.debug_info:
            answer = self.generator.generate(question, result)
            answer.retriever_name = self.name
            answer.debug_info["retrieval"] = result.debug_info
            answer.latency = time.perf_counter() - started
            return answer
        llm_mode = self.settings.get("gap_repair.answer_mode", "deterministic") == "llm"
        initial_temporary = [Relation.from_dict({**r, "id": "query-repair-" + str(i),
                                                "type": r["relation"], "attributes": {"temporary": True}})
                             for i, r in enumerate(result.debug_info["temporary_relations"])]
        context = ContextBuilder(self.settings).build(
            replace(result, relations=result.relations + initial_temporary) if llm_mode else result)
        used = {c.chunk_id for c in context.citations}
        debug = result.debug_info
        # ContextBuilder may truncate by characters as well as top_k. Do not claim
        # repairs whose source would be invisible to the answer generator/user.
        debug["temporary_relations"] = [r for r in debug["temporary_relations"]
                                        if any(e["chunk_id"] in used for e in r["evidences"])]
        for relation in debug["temporary_relations"]:
            relation["evidences"] = [e for e in relation["evidences"] if e["chunk_id"] in used]
            relation["supporting_documents"] = sorted({e["doc_id"] for e in relation["evidences"]})
        debug["compensation_documents"] = sorted(set(debug["compensation_documents"]) &
                                                   {c.doc_id for c in context.citations})
        edge_sources = {(r.head_id, r.type, r.tail_id): {e.chunk_id for e in r.evidences}
                        for r in result.relations}
        edge_sources.update({(r["head_id"], r["relation"], r["tail_id"]):
                             {e["chunk_id"] for e in r["evidences"]}
                             for r in debug["temporary_relations"]})
        answers = [a for a in debug["answer_ids"] if all(
            tuple(edge) in edge_sources and (not edge_sources[tuple(edge)] or edge_sources[tuple(edge)] & used)
            for edge in debug["answer_paths"][a])]
        debug["answer_ids"] = answers
        debug["context_truncated"] = context.truncated
        debug["status"] = "supported_answers" if answers else "unresolved"
        graph = Subgraph(result.entities, result.relations, result.scores,
                         list(debug["plan"]["subjects"]),
                         [n for n in result.evidence_nodes if n.chunk.id in used])
        if llm_mode:
            temporary = [Relation.from_dict({**r, "id": "query-repair-" + str(i),
                                             "type": r["relation"], "attributes": {"temporary": True}})
                         for i, r in enumerate(debug["temporary_relations"])]
            answer = self.generator.generate(question, replace(result, relations=result.relations + temporary))
            # Visible graph metrics must not count repaired edges as originally visible.
            answer.subgraph = graph
        else:
            names = {e.id: e.name for e in result.entities}
            lines = []
            if answers:
                for entity_id in answers:
                    sources = set().union(*(edge_sources[tuple(e)] for e in debug["answer_paths"][entity_id]))
                    marks = "".join(f"[{c.marker}]" for c in context.citations if c.chunk_id in sources)
                    name = names[entity_id]
                    if debug["plan"]["target_type"] == "Movie":
                        name = f"《{name}》"
                    lines.append(name + marks)
                prefix = "当前语料证据支持的答案：" if debug.get("evidence_tiers", {}).get("dataset_assertion") else "检索证据支持的答案："
                text = ("有（限当前语料）。" if debug["plan"]["boolean"] else prefix) + "、".join(lines) + "。"
                if debug["budget_exhausted"] or context.truncated:
                    text += "上下文预算不足，以上仅为有完整引用支持的部分结果。"
            else:
                assessment = debug.get("negative_assessment")
                if assessment and set(assessment["source_chunk_ids"]) <= used:
                    marks = "".join(f"[{c.marker}]" for c in context.citations)
                    text = "经图路径与双方片单交叉核查，未发现共同出演的影片（限本次检索到的当前语料证据）" + marks + "。"
                    debug["status"] = "no_support_after_corpus_audit"
                else:
                    text = "当前证据不足，尚未找到满足全部条件的完整路径，无法确定是否存在共同答案。"
            answer = Answer(text, context.citations, graph,
                            debug_info={"generator": "gap_repair_deterministic"})
        answer.retriever_name = self.name
        answer.latency = time.perf_counter() - started
        answer.debug_info["retrieval"] = debug
        return answer
