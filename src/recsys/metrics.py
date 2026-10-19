"""Ranking metrics for leave-last-out evaluation (one held-out item per user, full catalogue ranking)."""

import numpy as np


def hit_and_ndcg(ranked: np.ndarray, targets: np.ndarray, k: int = 10) -> tuple[float, float]:
    """ranked: (n_users, >=k) item indexes in rank order; targets: (n_users,)."""
    top = ranked[:, :k]
    hits = top == targets[:, None]
    hit = hits.any(axis=1)
    pos = hits.argmax(axis=1)
    ndcg = np.where(hit, 1.0 / np.log2(pos + 2), 0.0)
    return float(hit.mean()), float(ndcg.mean())


def coverage(ranked: np.ndarray, n_items: int, k: int = 10) -> float:
    """Share of the catalogue that appears in anyone's top-k."""
    return len(np.unique(ranked[:, :k])) / (n_items - 1)


def summarize(ranked: np.ndarray, targets: np.ndarray, n_items: int) -> dict[str, float]:
    r10, n10 = hit_and_ndcg(ranked, targets, 10)
    r50, _ = hit_and_ndcg(ranked, targets, 50) if ranked.shape[1] >= 50 else (float("nan"), None)
    return {"recall@10": r10, "ndcg@10": n10, "recall@50": r50, "coverage@10": coverage(ranked, n_items)}
