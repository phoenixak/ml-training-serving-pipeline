"""Baselines on the same split: global popularity and item-kNN (cosine over co-likes)."""

import numpy as np
import scipy.sparse as sp

from recsys.data import Data


def _mask_seen(scores: np.ndarray, histories: list[list[int]]) -> None:
    for row, h in enumerate(histories):
        scores[row, h] = -np.inf
    scores[:, 0] = -np.inf


def popularity(data: Data, histories: list[list[int]], k: int = 100) -> np.ndarray:
    pop = data.popularity()
    scores = np.tile(pop, (len(histories), 1))
    _mask_seen(scores, histories)
    return np.argsort(-scores, axis=1)[:, :k]


class ItemKNN:
    """Score = sum of cosine similarities between a candidate and the user's last `recent` likes."""

    def __init__(self, data: Data, recent: int = 50) -> None:
        rows, cols = [], []
        for u, h in data.train.items():
            rows.extend([u] * len(h))
            cols.extend(h)
        ui = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(data.train), data.n_items))
        norms = np.sqrt(np.asarray(ui.multiply(ui).sum(axis=0))).ravel() + 1e-9
        sim = (ui.T @ ui).tocsr().astype(np.float32)
        inv = sp.diags(1.0 / norms)
        self.sim = (inv @ sim @ inv).tocsr()
        self.sim.setdiag(0)
        self.recent = recent

    def recommend(self, histories: list[list[int]], k: int = 100, batch: int = 2048) -> np.ndarray:
        out = []
        for s in range(0, len(histories), batch):
            part = histories[s : s + batch]
            rows = [i for i, h in enumerate(part) for _ in h[-self.recent :]]
            cols = [j for h in part for j in h[-self.recent :]]
            q = sp.csr_matrix(
                (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(part), self.sim.shape[0])
            )
            scores = np.asarray((q @ self.sim).todense())
            _mask_seen(scores, part)
            out.append(np.argsort(-scores, axis=1)[:, :k])
        return np.vstack(out)
