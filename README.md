# Two-stage movie recommender — v2

[![CI](https://github.com/phoenixak/ml-training-serving-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/phoenixak/ml-training-serving-pipeline/actions/workflows/ci.yml)
[![Live demo](https://img.shields.io/badge/%F0%9F%A4%97%20Live%20demo-Hugging%20Face-F4F1EA?labelColor=141310)](https://huggingface.co/spaces/Phoenixak99/movie-recommender)
![Python](https://img.shields.io/badge/python-3.11%2B-141310)

The recommendation architecture large platforms use, trained end to end on **MovieLens-32M** and served without PyTorch: a **two-tower** model retrieves 100 candidates from the catalogue, a **DLRM-style ranker** re-orders them, and both run in **ONNX Runtime** in about 2 ms per request on two CPU threads.

**[Try the live demo →](https://huggingface.co/spaces/Phoenixak99/movie-recommender)** Pick a few movies you like; every recommendation says which of your picks it is closest to. Because users are represented by the movies they liked rather than by an id, a brand-new visitor gets real recommendations immediately.

## Results

Chronological leave-last-out on 20,000 held-out users: each user's last like is the target, ranked against the full catalogue of 4,779 movies with their earlier likes excluded.

| Method | Recall@10 | NDCG@10 | Recall@50 | Catalogue coverage@10 |
|---|---:|---:|---:|---:|
| Popularity | 0.040 | 0.020 | 0.129 | 2% |
| Item-kNN (cosine over co-likes) | 0.073 | 0.037 | 0.214 | 22% |
| Two-tower retrieval | 0.139 | 0.074 | 0.361 | 73% |
| **Two-tower + DLRM ranker** | **0.151** | **0.079** | **0.372** | **72%** |

The final system finds the held-out movie in its top 10 for **15.1%** of users — **3.7×** popularity and **2.1×** item-kNN — while still recommending from 72% of the catalogue rather than the same blockbusters.

**What didn't work first:** a ranker trained point-wise (binary cross-entropy per candidate, from scratch) *lowered* Recall@10 to 0.126 and collapsed coverage to 47%, drifting back toward popular movies. Making it a **residual** on the retrieval score (it starts exactly at the retrieval order and learns a correction) and training it **listwise** (a softmax over each user's 100 candidates) turned that into the +8% above.

## How it works

```
MovieLens-32M ratings ── rating ≥ 4 = like ──▶ 80,000 users · 4,779 movies · 7.3M training likes
                                                 last like = test · second-to-last = validation

 user's likes (≤50, newest first)                              every movie
          │                                                         │
   item tower (shared) ──▶ self-attention ──▶ user vector    item tower: id + genres + decade
          │                                         │               │
          └──────────────── dot product ◀───────────┴── 4,779 item vectors (precomputed)
                                    │
                         top-100 candidates (retrieval)
                                    │
        DLRM ranker: item id · genres · decade · user vec · item vec · dense features
        (retrieval score, popularity, genre affinity, similarity to recent likes …)
        → pairwise dot-product interactions → MLP → correction to the retrieval score
                                    │
                    top-10, each with "because you liked …"
```

| Piece | Details |
|---|---|
| Retrieval training | In-batch sampled softmax (batch 1,024, temperature 0.05) with logQ correction for popularity; 8 next-like samples per user per epoch; best epoch picked on validation NDCG@10 |
| User tower | One pre-norm self-attention block over the item vectors of the last 50 likes + recency embeddings, mean-pooled. Written out explicitly so PyTorch and ONNX run identical maths (tested) |
| Ranker | DLRM-style: 6 feature vectors projected to 16-d, all 15 pairwise dot products + dense features → MLP; residual on the retrieval logit; listwise loss on validation-time candidates |
| Tracking | MLflow (local store): parameters, per-epoch validation metrics and the final test table |
| Serving | `user_tower.onnx` + `ranker.onnx` + item vectors in numpy; no PyTorch at inference. Requests take ~2 ms on 2 threads |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[train,api,demo,dev]"

# data: MovieLens-32M from GroupLens (239 MB)
curl -L -o data/raw/ml-32m.zip https://files.grouplens.org/datasets/movielens/ml-32m.zip && unzip data/raw/ml-32m.zip -d data/raw
python -m recsys.data                     # → data/processed (~20 s)
python -m recsys.train --out artifacts    # ~25 min on a 16-thread CPU; evaluates, logs to MLflow, exports ONNX
mlflow ui --backend-store-uri ./mlruns    # browse runs

python space/app.py                       # the demo on http://127.0.0.1:7860
uvicorn recsys.api:app --app-dir src      # the API
pytest -q                                 # 7 tests, incl. ONNX ↔ PyTorch equivalence; no download needed
```

The exported serving files are committed in `artifacts/` (4.5 MB), so the API and demo work without training.

## API

| Endpoint | What it does |
|---|---|
| `POST /recommend` `{"liked": [2571, 589], "k": 10, "rerank": true}` | Top-k for a list of liked MovieLens `movieId`s (oldest first), each with `because` |
| `GET /similar/{movieId}` | Nearest movies in the item-vector space |
| `GET /movies/search?q=matrix` | Title search, most-liked first |
| `GET /health` | Status and catalogue size |

`docker build -t recsys . && docker run -p 8000:8000 recsys` serves it (onnxruntime + numpy only).

## What changed from v1

v1 (tagged [`v1`](https://github.com/phoenixak/ml-training-serving-pipeline/tree/v1)) trained a simplified DLRM on synthetic data, reported no evaluation and had no demo; its optional Azure ML job submission is not carried over. v2 trains on real data with a chronological split, compares against baselines, adds retrieval, a ranker that measurably helps, ONNX serving, tests that check the export is exact, and a public demo.

## Data and licence

Code: MIT. Data: **MovieLens 32M** — F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. The files in `artifacts/` are derived from it and are shared under the same GroupLens terms: research and non-commercial use, acknowledgement required, no endorsement by the University of Minnesota or GroupLens implied.
