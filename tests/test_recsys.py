import numpy as np
import pandas as pd
import pytest
import torch

from recsys.baselines import ItemKNN, popularity
from recsys.data import DECADES, GENRES, Data, clean_title, parse_year
from recsys.metrics import coverage, hit_and_ndcg
from recsys.model import MAX_HISTORY, DLRMRanker, TwoTower, in_batch_softmax_loss
from recsys.train import pad_newest_first, ranker_features, retrieve


def test_titles_and_years():
    assert parse_year("Matrix, The (1999)") == 1999.0
    assert clean_title("Matrix, The (1999)") == "The Matrix"
    assert (
        clean_title("Spirited Away (Sen to Chihiro no kamikakushi) (2001)")
        == "Spirited Away (Sen to Chihiro no kamikakushi)"
    )
    assert parse_year("Untitled") is None


def test_hit_and_ndcg():
    ranked = np.array([[5, 3, 9], [1, 2, 4], [7, 8, 6]])
    hit, ndcg = hit_and_ndcg(ranked, np.array([5, 4, 0]), k=3)
    assert hit == pytest.approx(2 / 3)
    assert ndcg == pytest.approx((1.0 + 1 / np.log2(4)) / 3)
    assert coverage(ranked, n_items=10, k=1) == pytest.approx(3 / 9)


def test_pad_newest_first():
    h = pad_newest_first([[1, 2, 3], list(range(1, 80))])
    assert h.shape == (2, MAX_HISTORY)
    assert h[0, :4].tolist() == [3, 2, 1, 0]
    assert h[1, 0].item() == 79 and h[1, -1].item() == 30


def test_in_batch_loss_ignores_duplicate_targets():
    u = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    v = u.clone()
    t = torch.tensor([1, 1, 2, 3])
    loss = in_batch_softmax_loss(u, v, t, torch.zeros(5), temperature=0.05)
    assert torch.isfinite(loss) and loss.item() < 1.0  # perfect alignment, duplicates masked


@pytest.fixture(scope="module")
def tiny():
    rng = np.random.default_rng(0)
    n = 40
    items = pd.DataFrame(
        {
            "idx": np.arange(1, n + 1),
            "movieId": np.arange(100, 100 + n),
            "tmdbId": np.arange(n),
            "title": [f"Movie {i}" for i in range(n)],
            "year": rng.integers(1950, 2024, n).astype(float),
            "genres": [[GENRES[i % len(GENRES)], GENRES[(i * 7) % len(GENRES)]] for i in range(n)],
            "n_likes": rng.integers(1, 100, n),
        }
    )
    train = {u: rng.choice(np.arange(1, n + 1), 12, replace=False).tolist() for u in range(60)}
    data = Data(
        items, train, {u: int(rng.integers(1, n + 1)) for u in train}, {u: int(rng.integers(1, n + 1)) for u in train}
    )
    model = TwoTower(torch.tensor(data.genre_matrix()), torch.tensor(data.decade_index()), len(DECADES), dim=16)
    ranker = DLRMRanker(data.n_items, len(GENRES), len(DECADES), tower_dim=16)
    return data, model, ranker


def test_baselines_exclude_history(tiny):
    data, _, _ = tiny
    hist = [data.train[0], data.train[1]]
    for ranked in (popularity(data, hist, k=10), ItemKNN(data).recommend(hist, k=10)):
        for row, h in zip(ranked, hist):
            assert not set(row) & set(h) and 0 not in row


def test_retrieve_and_ranker_shapes(tiny):
    data, model, ranker = tiny
    hist = [data.train[u] for u in range(5)]
    cand, scores, users, items = retrieve(model, hist, k=10)
    assert cand.shape == (5, 10) and users.shape == (5, 16)
    assert all(not set(c.tolist()) & set(h) for c, h in zip(cand, hist))
    feats = ranker_features(
        torch.tensor(data.genre_matrix()),
        torch.tensor(data.decade_index()),
        torch.tensor(np.log(data.popularity()), dtype=torch.float32),
        items,
        hist,
        cand,
        scores,
        users,
    )
    assert ranker(**feats).shape == (50,)


def test_onnx_serving_matches_torch(tiny, tmp_path):
    pytest.importorskip("onnxruntime")
    from recsys.export import export
    from recsys.serve import Recommender

    data, model, ranker = tiny
    export(model, ranker, data, tmp_path)
    rec = Recommender(tmp_path, threads=1)
    history = data.train[3]
    with torch.no_grad():
        expected = model.user_vectors(pad_newest_first([history]))[0].numpy()
    np.testing.assert_allclose(rec.user_vector(history), expected, atol=1e-5)

    liked = [int(data.items.loc[data.items["idx"] == i, "movieId"].iloc[0]) for i in history]
    out = rec.recommend(liked, k=5)
    assert len(out) == 5 and not {o["idx"] for o in out} & set(history)
    assert all(o["because"] in set(data.items["title"]) for o in out)
    assert rec.similar(liked[0], k=3)[0]["idx"] != history[0]
    assert rec.search("movie 1")[0]["title"].startswith("Movie 1")
