"""
This module benchmarks retrieval quality for predefined task groups.
It evaluates the hybrid search stage only (without generation/reranking)
against atom-level ground truth labels and reports standard IR metrics.
"""

import json
from math import log2
from pathlib import Path

import yaml
from app.core.config import get_rag_settings
from app.infrastructure.repositories.task_repo import normalize_task_text
from app.infrastructure.retrieval.engine import RagEngine, _sparse_to_qdrant
from app.infrastructure.vector.embeddings import embed_query, embed_query_sparse

K_VALUES = (1, 3, 5, 10)
TASK_NUMBERS = (6, 10, 12)
GT_FILES = {
    6: Path("app/data/atoms_t6_applicable_tasks.yaml"),
    10: Path("app/data/atoms_t10_applicable_tasks.yaml"),
    12: Path("app/data/atoms_t12_applicable_tasks.yaml"),
}
TASKS_FILE = Path("fipi_parsed_katex_with_solutions.json")


def load_ground_truth(task_number: int) -> dict[str, set[str]]:
    """
    Build mapping: task_id -> set of relevant atom IDs.
    The YAML files store atoms and their "applicable_tasks", so we invert
    that structure for fast relevance lookups during evaluation.
    """
    atoms = yaml.safe_load(GT_FILES[task_number].read_text(encoding="utf-8")) or []
    gt: dict[str, set[str]] = {}
    for atom in atoms:
        atom_id = str(atom.get("id", "")).strip()
        if not atom_id:
            continue
        for raw_task_id in atom.get("applicable_tasks", []):
            task_id = str(raw_task_id).strip()
            if not task_id:
                continue
            gt.setdefault(task_id, set()).add(atom_id)
    return gt


def dcg_at_k(binary_rels: list[int], k: int) -> float:
    """
    Discounted Cumulative Gain rewards relevant hits at higher ranks.
    Using log2 discount keeps rank-1 most valuable and smoothly decays after.
    """
    # Using log2 discount keeps rank-1 most valuable and smoothly decays after.
    score = 0.0
    for i, rel in enumerate(binary_rels[:k], start=1):
        if rel:
            score += 1 / log2(i + 1)
    return score


def precision_at_k(binary_rels: list[int], k: int) -> float:
    """
    Share of retrieved items in top-k that are relevant.
    """
    return sum(binary_rels[:k]) / k


def recall_at_k(binary_rels: list[int], relevant_count: int, k: int) -> float:
    """
    Fraction of all relevant items recovered by top-k results.
    """
    if relevant_count == 0:
        return 0.0
    return sum(binary_rels[:k]) / relevant_count


def hitrate_at_k(binary_rels: list[int], k: int) -> float:
    """
    Binary success metric: did we retrieve at least one relevant item in top-k?
    """
    return 1.0 if any(binary_rels[:k]) else 0.0


def mrr_at_k(binary_rels: list[int], k: int) -> float:
    """
    Reciprocal rank of the first relevant result in top-k.
    Returns 0.0 when no relevant result is present.
    """
    # Returns 0.0 when no relevant result is present.
    for rank, rel in enumerate(binary_rels[:k], start=1):
        if rel:
            return 1.0 / rank
    return 0.0


def average_precision_at_k(binary_rels: list[int], relevant_count: int, k: int) -> float:
    """
    Average Precision aggregates precision at each relevant hit position.
    Denominator is clipped by k to make AP@k comparable across queries.
    """
    if relevant_count == 0:
        return 0.0
    hit_count = 0
    acc = 0.0
    for rank, rel in enumerate(binary_rels[:k], start=1):
        if rel:
            hit_count += 1
            acc += hit_count / rank
    denom = min(relevant_count, k)
    return acc / denom if denom > 0 else 0.0


def ndcg_at_k(binary_rels: list[int], relevant_count: int, k: int) -> float:
    # NDCG normalizes DCG by the ideal ranking for this query, yielding [0, 1].
    dcg = dcg_at_k(binary_rels, k)
    ideal_rels = [1] * min(relevant_count, k)
    idcg = dcg_at_k(ideal_rels, k)
    return dcg / idcg if idcg > 0 else 0.0


def empty_metric_bucket() -> dict[int, dict[str, float]]:
    """
    Pre-allocate metric accumulators for each k.
    We sum per-query metric values and divide by evaluated query count later.
    """
    # We sum per-query metric values and divide by evaluated query count later.
    return {
        k: {
            "precision": 0.0,
            "recall": 0.0,
            "ndcg": 0.0,
            "mrr": 0.0,
            "map": 0.0,
            "hitrate": 0.0,
        }
        for k in K_VALUES
    }


def run_hybrid_search_only(rag: RagEngine, query: str, task_number: int, limit: int) -> list[str]:
    """
    Execute the same hybrid retrieval stack used in production:
    dense embedding + optional sparse/ColBERT + weighted reciprocal rank fusion.
    We bypass answer generation to isolate retrieval quality.
    """
    cfg = get_rag_settings()
    q_vec = embed_query(query)
    q_sparse = _sparse_to_qdrant(embed_query_sparse(query)) if cfg.bm25_enabled else None
    hits = rag._store.hybrid_search(  # noqa: SLF001
        q_vec,
        q_sparse,
        query_text=query if cfg.reranker_enabled else None,
        task_number=task_number,
        subtypes=None,
        limit=limit,
        prefetch_factor=cfg.qdrant_prefetch_factor,
        dense_weight=cfg.dense_weight,
        bm25_weight=cfg.bm25_weight,
        rrf_k=cfg.rrf_k,
    )
    return [hit.doc.doc_id for hit in hits]


def buckets_sums_to_macro_averages(
    buckets: dict[int, dict[str, float]], evaluated: int
) -> dict[str, dict[str, float]]:
    """Convert accumulated sums to macro-averaged metrics (same as print_metrics)."""
    if evaluated == 0:
        return {str(k): {m: 0.0 for m in buckets[k]} for k in K_VALUES}
    return {str(k): {m: buckets[k][m] / evaluated for m in buckets[k]} for k in K_VALUES}


def buckets_to_jsonable(
    buckets: dict[int, dict[str, float]],
) -> dict[str, dict[str, float]]:
    return {str(k): dict(buckets[k]) for k in K_VALUES}


def print_metrics(title: str, buckets: dict[int, dict[str, float]], evaluated: int) -> None:
    """
    Convert accumulated sums into macro-averaged metrics over queries.
    """
    print(f"\n{title}: evaluated={evaluated}")
    if evaluated == 0:
        print("No samples with ground truth intersection found.")
        return
    for k in K_VALUES:
        p = buckets[k]["precision"] / evaluated
        r = buckets[k]["recall"] / evaluated
        n = buckets[k]["ndcg"] / evaluated
        mrr = buckets[k]["mrr"] / evaluated
        ap = buckets[k]["map"] / evaluated
        hr = buckets[k]["hitrate"] / evaluated
        print(
            f"@{k}: "
            f"precision={p:.4f} "
            f"recall={r:.4f} "
            f"ndcg={n:.4f} "
            f"mrr={mrr:.4f} "
            f"map={ap:.4f} "
            f"hitrate={hr:.4f}"
        )


def evaluate_task_number(
    rag: RagEngine, task_number: int
) -> tuple[dict[int, dict[str, float]], int]:
    """
    Evaluate one task group (e.g. 6/10/12):
    1. load relevance labels,
    2. run retrieval for each query,
    3. accumulate metrics for all configured k values.
    """
    gt = load_ground_truth(task_number)
    data = json.loads(TASKS_FILE.read_text(encoding="utf-8"))

    buckets = empty_metric_bucket()
    evaluated = 0
    max_k = max(K_VALUES)

    for item in data:
        if item.get("task_number") != task_number:
            continue
        task_id = str(item.get("id", "")).strip()
        relevant = gt.get(task_id)
        if not relevant:
            continue
        raw_text = (item.get("problem_katex") or item.get("question") or "").strip()
        query = normalize_task_text(raw_text)
        if not query:
            continue

        # Retrieve top-N candidates and mark relevance against ground truth.
        ranked_ids = run_hybrid_search_only(rag, query, task_number, limit=max_k)
        binary_rels = [1 if atom_id in relevant else 0 for atom_id in ranked_ids]
        evaluated += 1
        rel_count = len(relevant)

        for k in K_VALUES:
            buckets[k]["precision"] += precision_at_k(binary_rels, k)
            buckets[k]["recall"] += recall_at_k(binary_rels, rel_count, k)
            buckets[k]["ndcg"] += ndcg_at_k(binary_rels, rel_count, k)
            buckets[k]["mrr"] += mrr_at_k(binary_rels, k)
            buckets[k]["map"] += average_precision_at_k(binary_rels, rel_count, k)
            buckets[k]["hitrate"] += hitrate_at_k(binary_rels, k)

    print_metrics(f"Task {task_number}", buckets, evaluated)
    return buckets, evaluated


def run_benchmark() -> dict:
    """
    Run per-task and aggregate evaluation; return serializable result dict.
    """
    rag = RagEngine()
    total_buckets = empty_metric_bucket()
    total_evaluated = 0
    per_task: dict = {}
    for task_number in TASK_NUMBERS:
        buckets, evaluated = evaluate_task_number(rag, task_number)
        per_task[str(task_number)] = {
            "evaluated": evaluated,
            "bucket_sums": buckets_to_jsonable(buckets),
            "macro_averages": buckets_sums_to_macro_averages(buckets, evaluated),
        }
        total_evaluated += evaluated
        for k in K_VALUES:
            for metric in total_buckets[k]:
                total_buckets[k][metric] += buckets[k][metric]
    print_metrics("All tasks", total_buckets, total_evaluated)
    return {
        "k_values": list(K_VALUES),
        "task_numbers": list(TASK_NUMBERS),
        "per_task": per_task,
        "all_tasks": {
            "evaluated": total_evaluated,
            "bucket_sums": buckets_to_jsonable(total_buckets),
            "macro_averages": buckets_sums_to_macro_averages(total_buckets, total_evaluated),
        },
    }


def main() -> None:
    """
    Run per-task evaluation and then print aggregate scores across all tasks.
    """
    run_benchmark()


if __name__ == "__main__":
    main()
