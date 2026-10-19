"""MovieLens-32M → implicit "liked" sequences with a chronological leave-last-out split.

A rating of 4 or more counts as a like. Each user's likes are ordered by time: the last one is the
test target, the one before it the validation target, and everything earlier is training history.

    python -m recsys.data --raw data/raw/ml-32m --out data/processed
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "IMAX",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]
DECADES = list(range(1900, 2030, 10))  # bucket index = (year - 1900) // 10, clipped
PAD = 0  # item index 0 is padding; real items start at 1


@dataclass
class Data:
    items: pd.DataFrame  # idx, movieId, tmdbId, title, year, genres (list), n_likes
    train: dict[int, list[int]]  # user → chronological item indexes (history)
    val: dict[int, int]  # user → next item after train
    test: dict[int, int]  # user → item after val

    @property
    def n_items(self) -> int:
        return len(self.items) + 1  # + padding

    def genre_matrix(self) -> np.ndarray:
        g = np.zeros((self.n_items, len(GENRES)), dtype=np.float32)
        for idx, genres in zip(self.items["idx"], self.items["genres"]):
            for name in genres:
                if name in GENRES:
                    g[idx, GENRES.index(name)] = 1.0
        return g

    def decade_index(self) -> np.ndarray:
        d = np.zeros(self.n_items, dtype=np.int64)
        years = self.items["year"].fillna(1990).astype(int).to_numpy()
        d[self.items["idx"].to_numpy()] = np.clip((years - 1900) // 10, 0, len(DECADES) - 1)
        return d

    def popularity(self) -> np.ndarray:
        """Share of training likes per item (index 0 = padding gets the smallest share)."""
        counts = np.bincount(np.concatenate([np.array(h) for h in self.train.values()]), minlength=self.n_items).astype(
            np.float64
        )
        counts[PAD] = 0
        return (counts + 1) / (counts.sum() + self.n_items)


def parse_year(title: str) -> float | None:
    m = re.search(r"\((\d{4})\)\s*$", title.strip())
    return float(m.group(1)) if m else None


def clean_title(title: str) -> str:
    t = re.sub(r"\s*\(\d{4}\)\s*$", "", title.strip())
    m = re.match(r"^(.*), (The|A|An|Les|La|Le|Il|El|Die|Das)$", t)  # "Matrix, The" → "The Matrix"
    return f"{m.group(2)} {m.group(1)}" if m else t


def build(
    raw: Path, min_item_likes: int = 300, min_user_likes: int = 20, max_users: int = 80_000, seed: int = 0
) -> Data:
    ratings = pd.read_csv(
        raw / "ratings.csv", dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"}
    )
    likes = ratings[ratings["rating"] >= 4.0]
    item_counts = likes["movieId"].value_counts()
    likes = likes[likes["movieId"].isin(item_counts[item_counts >= min_item_likes].index)]
    user_counts = likes["userId"].value_counts()
    users = user_counts[user_counts >= min_user_likes].index.to_numpy()
    rng = np.random.default_rng(seed)
    if len(users) > max_users:
        users = rng.choice(users, max_users, replace=False)
    likes = likes[likes["userId"].isin(users)].sort_values(["userId", "timestamp", "movieId"])

    movies = pd.read_csv(raw / "movies.csv")
    links = pd.read_csv(raw / "links.csv")
    kept = likes["movieId"].value_counts()
    items = (
        movies[movies["movieId"].isin(kept.index)]
        .merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
        .assign(n_likes=lambda d: d["movieId"].map(kept).astype(int))
        .sort_values("n_likes", ascending=False)
        .reset_index(drop=True)
    )
    items["idx"] = np.arange(1, len(items) + 1)
    items["year"] = items["title"].map(parse_year)
    items["title"] = items["title"].map(clean_title)
    items["genres"] = items["genres"].str.split("|")
    to_idx = dict(zip(items["movieId"], items["idx"]))

    train, val, test = {}, {}, {}
    for u, (_uid, seq) in enumerate(likes.groupby("userId", sort=False)["movieId"]):
        s = [to_idx[m] for m in seq.to_numpy()]
        train[u], val[u], test[u] = s[:-2], s[-2], s[-1]
    return Data(items[["idx", "movieId", "tmdbId", "title", "year", "genres", "n_likes"]], train, val, test)


def save(d: Data, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    d.items.to_parquet(out / "items.parquet")
    users = sorted(d.train)
    np.savez_compressed(
        out / "sequences.npz",
        lengths=np.array([len(d.train[u]) for u in users]),
        history=np.concatenate([np.array(d.train[u], dtype=np.int32) for u in users]),
        val=np.array([d.val[u] for u in users]),
        test=np.array([d.test[u] for u in users]),
    )
    stats = {
        "users": len(users),
        "items": len(d.items),
        "train_likes": int(sum(len(h) for h in d.train.values())),
        "median_history": float(np.median([len(h) for h in d.train.values()])),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))


def load(out: Path) -> Data:
    items = pd.read_parquet(out / "items.parquet")
    items["genres"] = items["genres"].map(list)
    z = np.load(out / "sequences.npz")
    ends = np.cumsum(z["lengths"])
    starts = ends - z["lengths"]
    hist = z["history"]
    train = {u: hist[s:e].tolist() for u, (s, e) in enumerate(zip(starts, ends))}
    return Data(items, train, dict(enumerate(z["val"].tolist())), dict(enumerate(z["test"].tolist())))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/ml-32m")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--max-users", type=int, default=80_000)
    args = ap.parse_args()
    data = build(Path(args.raw), max_users=args.max_users)
    save(data, Path(args.out))
    print((Path(args.out) / "stats.json").read_text())
