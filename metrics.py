"""Retrieval metrics for the ToolRet corpus-scaling evaluation.

Binary relevance throughout: a tool is either a ground-truth target for a
query or it is not. ToolRet ships labels in exactly that form.

nDCG@10 is the benchmark's headline metric, so it is the primary number
here; Recall@10 is reported alongside because it answers the question an
agent builder actually cares about -- "did the right tool make the
shortlist at all?"
"""
from __future__ import annotations

import math
import statistics


def dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked_ids: list[str], relevant: set[str], k: int = 10) -> float:
    """Normalised DCG with binary gains.

    The ideal ranking puts min(k, |relevant|) hits at the top, so a query
    with more targets than k is not penalised for the impossible.
    """
    if not relevant:
        return 0.0
    gains = [1.0 if tid in relevant else 0.0 for tid in ranked_ids[:k]]
    ideal = [1.0] * min(k, len(relevant))
    denom = dcg(ideal)
    return dcg(gains) / denom if denom else 0.0


def recall_at_k(ranked_ids: list[str], relevant: set[str], k: int = 10) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def completeness_at_k(ranked_ids: list[str], relevant: set[str], k: int = 10) -> float:
    """1.0 only if *every* target for the query is in the top k.

    ToolRet reports this because many tasks need a full set of tools, not
    just one -- getting 2 of 3 right still leaves the agent unable to act.
    """
    if not relevant:
        return 0.0
    return 1.0 if relevant <= set(ranked_ids[:k]) else 0.0


def evaluate(
    run: dict[str, list[str]], qrels: dict[str, dict[str, int]], k: int = 10
) -> dict[str, float]:
    """Aggregate metrics over a run.

    `run` maps query id -> ranked tool ids. `qrels` maps query id -> {tool
    id: relevance}. Queries present in qrels but missing from the run score
    zero rather than being skipped, so a pipeline cannot improve its score
    by declining to answer.
    """
    per_query = {"ndcg": [], "recall": [], "completeness": []}
    for qid, rels in qrels.items():
        relevant = {t for t, r in rels.items() if r > 0}
        ranked = run.get(qid, [])
        per_query["ndcg"].append(ndcg_at_k(ranked, relevant, k))
        per_query["recall"].append(recall_at_k(ranked, relevant, k))
        per_query["completeness"].append(completeness_at_k(ranked, relevant, k))

    return {
        f"ndcg@{k}": statistics.mean(per_query["ndcg"]),
        f"recall@{k}": statistics.mean(per_query["recall"]),
        f"completeness@{k}": statistics.mean(per_query["completeness"]),
        "n_queries": len(qrels),
        "n_answered": sum(1 for q in qrels if run.get(q)),
    }
