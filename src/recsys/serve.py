"""Inference without PyTorch: onnxruntime for the user tower and ranker, numpy for retrieval.

from recsys.serve import Recommender
rec = Recommender("artifacts")
rec.recommend(liked=[318, 2571, 79132])     # MovieLens movieIds (Shawshank, The Matrix, Inception)
"""

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import onnxruntime as ort


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]+", " ", s).strip()


class Recommender:
    def __init__(self, artifacts: str | Path, threads: int = 2) -> None:
        a = Path(artifacts)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.user_tower = ort.InferenceSession(str(a / "user_tower.onnx"), opts, providers=["CPUExecutionProvider"])
        self.ranker = ort.InferenceSession(str(a / "ranker.onnx"), opts, providers=["CPUExecutionProvider"])
        self.vectors = np.load(a / "item_vectors.npy")
        self.genres = np.load(a / "item_genres.npy")
        self.decades = np.load(a / "item_decades.npy")
        self.log_pop = np.load(a / "item_log_pop.npy")
        self.max_history = json.loads((a / "manifest.json").read_text())["max_history"]
        self.items = {m["idx"]: m for m in json.loads((a / "items.json").read_text(encoding="utf-8"))}
        self.by_movie_id = {m["movieId"]: idx for idx, m in self.items.items()}
        self._search = [(_norm(f"{m['title']} {m['year']}"), idx) for idx, m in self.items.items()]

    # ── lookup ────────────────────────────────────────────────────────────
    def search(self, query: str, limit: int = 20) -> list[dict]:
        q = _norm(query)
        if not q:
            return []
        hits = [(0 if t.startswith(q) else 1, -self.items[i]["n_likes"], i) for t, i in self._search if q in t]
        return [self.items[i] for *_, i in sorted(hits)[:limit]]

    def to_idx(self, movie_ids: list[int]) -> list[int]:
        return [self.by_movie_id[m] for m in movie_ids if m in self.by_movie_id]

    # ── models ────────────────────────────────────────────────────────────
    def user_vector(self, history: list[int]) -> np.ndarray:
        h = np.zeros((1, self.max_history), dtype=np.int64)
        recent = history[-self.max_history :][::-1]
        h[0, : len(recent)] = recent
        return self.user_tower.run(None, {"history": h})[0][0]

    def _rank(self, history: list[int], cand: np.ndarray, cand_scores: np.ndarray, user: np.ndarray) -> np.ndarray:
        k = len(cand)
        hist_genre = self.genres[history].mean(0)
        hv = self.vectors[history[-20:]]
        cv = self.vectors[cand]
        dense = np.stack(
            [
                cand_scores,
                self.log_pop[cand],
                (self.decades[cand] - 9) / 3,
                self.genres[cand] @ hist_genre,
                (cv @ hv.T).max(1),
                (cv @ hv[-5:].T).mean(1),
                np.full(k, np.log1p(len(history)) / 5),
            ],
            axis=1,
        ).astype(np.float32)
        feeds = {
            "item_idx": cand.astype(np.int64),
            "item_genres": self.genres[cand],
            "item_decade": self.decades[cand],
            "user_vec": np.repeat(user[None], k, 0),
            "item_vec": cv,
            "dense": dense,
        }
        return self.ranker.run(None, feeds)[0]

    # ── API ───────────────────────────────────────────────────────────────
    def recommend(self, liked: list[int], k: int = 10, candidates: int = 100, rerank: bool = True) -> list[dict]:
        """liked: MovieLens movieIds, oldest first. Returns the top-k with a 'because you liked' movie."""
        history = self.to_idx(liked)
        if not history:
            return []
        user = self.user_vector(history)
        scores = self.vectors @ user
        scores[0] = -np.inf
        scores[history] = -np.inf
        candidates = min(candidates, int(np.isfinite(scores).sum()))  # never more than the unseen catalogue
        if candidates == 0:
            return []
        cand = np.argpartition(-scores, candidates - 1)[:candidates]
        cand = cand[np.argsort(-scores[cand])]
        order = np.argsort(-self._rank(history, cand, scores[cand], user)) if rerank else np.arange(len(cand))
        top = cand[order[:k]]
        because = np.array(history)[(self.vectors[top] @ self.vectors[history].T).argmax(1)]
        return [
            {**self.items[int(i)], "score": float(scores[i]), "because": self.items[int(b)]["title"]}
            for i, b in zip(top, because)
        ]

    def similar(self, movie_id: int, k: int = 10) -> list[dict]:
        idx = self.by_movie_id.get(movie_id)
        if idx is None:
            return []
        sims = self.vectors @ self.vectors[idx]
        sims[[0, idx]] = -np.inf
        top = np.argsort(-sims)[:k]
        return [{**self.items[int(i)], "similarity": float(sims[i])} for i in top]
