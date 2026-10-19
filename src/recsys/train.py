"""Train retrieval + ranking, compare with baselines, log to MLflow and export for serving.

python -m recsys.data                    # once: MovieLens-32M → data/processed
python -m recsys.train --out artifacts   # train, evaluate, export ONNX + item vectors
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from recsys.baselines import ItemKNN, popularity
from recsys.data import DECADES, GENRES, Data, load
from recsys.metrics import summarize
from recsys.model import MAX_HISTORY, DLRMRanker, TwoTower, in_batch_softmax_loss

CANDIDATES = 100


def mlflow_name(metric: str) -> str:
    return metric.replace("@", "_at_")  # MLflow metric names can't contain "@"


def pad_newest_first(histories: list[list[int]], length: int = MAX_HISTORY) -> torch.Tensor:
    out = torch.zeros(len(histories), length, dtype=torch.long)
    for i, h in enumerate(histories):
        recent = h[-length:][::-1]
        out[i, : len(recent)] = torch.tensor(recent, dtype=torch.long)
    return out


def training_pairs(data: Data, per_user: int, rng: np.random.Generator) -> tuple[list[list[int]], np.ndarray]:
    """(history prefix, next like) pairs drawn from inside each user's training sequence."""
    hist, tgt = [], []
    for h in data.train.values():
        if len(h) < 2:
            continue
        for k in rng.integers(1, len(h), size=min(per_user, len(h) - 1)):
            hist.append(h[max(0, k - MAX_HISTORY) : k])
            tgt.append(h[k])
    return hist, np.array(tgt)


@torch.no_grad()
def retrieve(model: TwoTower, histories: list[list[int]], k: int = CANDIDATES, batch: int = 4096):
    model.eval()
    items = model.items.all_items()
    ids, scores, users = [], [], []
    for s in range(0, len(histories), batch):
        part = histories[s : s + batch]
        u = model.user_vectors(pad_newest_first(part))
        sc = u @ items.T
        sc[:, 0] = -torch.inf
        for row, h in enumerate(part):
            sc[row, h] = -torch.inf
        top = sc.topk(k, dim=1)
        ids.append(top.indices)
        scores.append(top.values)
        users.append(u)
    return torch.cat(ids), torch.cat(scores), torch.cat(users), items


def ranker_features(data_genres, decades, pop_log, items_vec, histories, cand, cand_scores, user_vecs):
    """Inputs for the ranker for every (user, candidate) pair, flattened to (users*k, …)."""
    n, k = cand.shape
    flat = cand.reshape(-1)
    hist_genre = torch.stack([data_genres[h].mean(0) if h else torch.zeros(data_genres.shape[1]) for h in histories])
    item_genre = data_genres[flat]
    affinity = (hist_genre.repeat_interleave(k, 0) * item_genre).sum(1)
    hv = [items_vec[h[-20:]] for h in histories]  # last 20 likes
    cv = items_vec[cand]  # (n, k, d)
    max_sim = torch.stack(
        [(cv[i] @ hv[i].T).max(1).values if len(hv[i]) else torch.zeros(k) for i in range(n)]
    ).reshape(-1)
    mean_recent = torch.stack(
        [(cv[i] @ hv[i][-5:].T).mean(1) if len(hv[i]) else torch.zeros(k) for i in range(n)]
    ).reshape(-1)
    hist_len = torch.tensor([np.log1p(len(h)) for h in histories], dtype=torch.float32).repeat_interleave(k)
    dense = torch.stack(
        [
            cand_scores.reshape(-1),
            pop_log[flat],
            (decades[flat].float() - 9) / 3,
            affinity,
            max_sim,
            mean_recent,
            hist_len / 5,
        ],
        dim=1,
    )
    return {
        "item_idx": flat,
        "item_genres": item_genre,
        "item_decade": decades[flat],
        "user_vec": user_vecs.repeat_interleave(k, 0),
        "item_vec": items_vec[flat],
        "dense": dense,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--per-user", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--ranker-users", type=int, default=30_000)
    ap.add_argument("--eval-users", type=int, default=20_000)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--reuse-retrieval", action="store_true", help="load the two-tower model from --out")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = load(Path(args.data))
    genres = torch.tensor(data.genre_matrix())
    decades = torch.tensor(data.decade_index())
    log_pop = torch.tensor(np.log(data.popularity()), dtype=torch.float32)
    users = sorted(data.train)
    eval_users = rng.choice(users, min(args.eval_users, len(users)), replace=False)
    print(f"{len(users)} users · {data.n_items - 1} items · {sum(len(h) for h in data.train.values())} training likes")

    run = None
    if not args.no_mlflow:
        import os

        # local file store: lightweight (mlflow-skinny), view with `mlflow ui --backend-store-uri ./mlruns`
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        import mlflow

        mlflow.set_tracking_uri(f"file:{Path('mlruns').resolve().as_posix()}")
        mlflow.set_experiment("two-stage-recommender")
        run = mlflow.start_run()
        mlflow.log_params(vars(args))

    # ── retrieval ────────────────────────────────────────────────────────
    model = TwoTower(genres, decades, len(DECADES), args.dim)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    val_hist = [data.train[u] for u in eval_users]
    val_tgt = np.array([data.val[u] for u in eval_users])
    best, best_state = -1.0, None
    for epoch in range(1, (0 if args.reuse_retrieval else args.epochs) + 1):
        model.train()
        hist, tgt = training_pairs(data, args.per_user, rng)
        order = rng.permutation(len(tgt))
        t0, total = time.time(), 0.0
        steps = 0
        for s in range(0, len(order), args.batch):
            idx = order[s : s + args.batch]
            h = pad_newest_first([hist[i] for i in idx])
            t = torch.tensor(tgt[idx])
            loss = in_batch_softmax_loss(model.user_vectors(h), model.items(t), t, log_pop, args.temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            steps += 1
        ranked, _, _, _ = retrieve(model, val_hist)
        m = summarize(ranked.numpy(), val_tgt, data.n_items)
        print(
            f"epoch {epoch}: loss {total / steps:.3f} · val recall@10 {m['recall@10']:.4f} ndcg@10 {m['ndcg@10']:.4f} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
        if run:
            mlflow.log_metrics(
                {"train_loss": total / steps, **{f"val_{mlflow_name(k)}": v for k, v in m.items()}}, step=epoch
            )
        if m["ndcg@10"] > best:
            best, best_state = m["ndcg@10"], {k: v.clone() for k, v in model.state_dict().items()}
    if args.reuse_retrieval:  # keep the towers from the last run, retrain only the ranker
        model.load_state_dict(torch.load(out / "checkpoint.pt", weights_only=True)["model"])
    else:
        model.load_state_dict(best_state)

    # ── ranker: trained on validation-time candidates (target = the validation like) ──
    rank_users = rng.choice(users, min(args.ranker_users, len(users)), replace=False)
    r_hist = [data.train[u] for u in rank_users]
    cand, cscore, uvec, items_vec = retrieve(model, r_hist)
    target = torch.tensor([data.val[u] for u in rank_users])
    keep = (cand == target[:, None]).any(1)
    print(f"ranker: validation target is in the top-{CANDIDATES} for {keep.float().mean():.1%} of users")
    sel = keep.nonzero().squeeze(1).tolist()
    feats = ranker_features(
        genres, decades, log_pop, items_vec, [r_hist[i] for i in sel], cand[sel], cscore[sel], uvec[sel]
    )
    # listwise: a softmax over each user's candidates, the correct one being the validation like
    pos = (cand[sel] == target[sel][:, None]).float().argmax(1)
    ranker = DLRMRanker(data.n_items, len(GENRES), len(DECADES), args.dim)
    ropt = torch.optim.AdamW(ranker.parameters(), lr=1e-3, weight_decay=1e-5)
    n_users, k = len(sel), cand.shape[1]
    for epoch in range(1, 5):
        perm = torch.randperm(n_users)
        total = 0.0
        for s in range(0, n_users, 256):
            users_b = perm[s : s + 256]
            rows = (users_b[:, None] * k + torch.arange(k)[None]).reshape(-1)
            logits = ranker(**{name: v[rows] for name, v in feats.items()}).reshape(len(users_b), k)
            loss = F.cross_entropy(logits, pos[users_b])
            ropt.zero_grad()
            loss.backward()
            ropt.step()
            total += loss.item() * len(users_b)
        print(f"ranker epoch {epoch}: listwise loss {total / n_users:.4f}", flush=True)

    # ── test: history = train + validation like, target = the last like ──
    test_hist = [data.train[u] + [data.val[u]] for u in eval_users]
    test_tgt = np.array([data.test[u] for u in eval_users])
    results = {}
    t0 = time.time()
    results["Popularity"] = summarize(popularity(data, test_hist), test_tgt, data.n_items)
    results["Item-kNN (cosine over co-likes)"] = summarize(ItemKNN(data).recommend(test_hist), test_tgt, data.n_items)
    cand, cscore, uvec, items_vec = retrieve(model, test_hist)
    results["Two-tower retrieval"] = summarize(cand.numpy(), test_tgt, data.n_items)
    ranker.eval()
    with torch.no_grad():
        scores = torch.cat(
            [
                ranker(
                    **ranker_features(
                        genres,
                        decades,
                        log_pop,
                        items_vec,
                        test_hist[s : s + 2000],
                        cand[s : s + 2000],
                        cscore[s : s + 2000],
                        uvec[s : s + 2000],
                    )
                )
                for s in range(0, len(test_hist), 2000)
            ]
        ).reshape(cand.shape)
    reranked = torch.gather(cand, 1, scores.argsort(dim=1, descending=True))
    results["Two-tower + DLRM ranker"] = summarize(reranked.numpy(), test_tgt, data.n_items)
    print(json.dumps(results, indent=2), f"\n(test evaluation {time.time() - t0:.0f}s)")

    torch.save({"model": model.state_dict(), "ranker": ranker.state_dict(), "dim": args.dim}, out / "checkpoint.pt")
    (out / "metrics.json").write_text(
        json.dumps({"test": results, "users_evaluated": len(eval_users), "params": vars(args)}, indent=2)
    )
    if run:
        for name, m in results.items():
            mlflow.log_metrics({f"test_{name.split()[0].lower()}_{mlflow_name(k)}": v for k, v in m.items() if v == v})
        mlflow.log_artifact(str(out / "metrics.json"))
        mlflow.end_run()

    from recsys.export import export

    export(model, ranker, data, out)


if __name__ == "__main__":
    main()
