"""Retrieval metrics. Empty input is undefined (NaN), never a fake 0 or 1."""
import numpy as np


def recall_at_k(ranked: dict, gold: dict, k: int) -> float:
    """Fraction of questions whose top-k ranked ids contain at least one gold id."""
    if not gold:
        return float("nan")
    return float(np.mean([bool(set(ranked.get(q, [])[:k]) & g) for q, g in gold.items()]))


def mrr(ranked: dict, gold: dict) -> float:
    """Mean reciprocal rank of the first gold id (0 if absent)."""
    if not gold:
        return float("nan")
    rr = []
    for q, g in gold.items():
        r = next((i + 1 for i, x in enumerate(ranked.get(q, [])) if x in g), None)
        rr.append(1 / r if r else 0.0)
    return float(np.mean(rr))
