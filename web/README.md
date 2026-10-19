---
title: Movie Recommender
emoji: 🎬
colorFrom: gray
colorTo: red
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Two-tower retrieval + DLRM ranking, in your browser
---

# Movie Recommender

Pick a few movies you like and get ten more, each with a "because you liked X" explanation. A two-tower model trained on MovieLens-32M retrieves 100 candidates from 4,779 movies, and a DLRM-style ranker re-orders them. Both models run in the browser with ONNX Runtime Web, so nothing is sent to a server. The results match the Python server (`recsys.serve`) exactly.

Code, training and evaluation: [phoenixak/ml-training-serving-pipeline](https://github.com/phoenixak/ml-training-serving-pipeline). This folder is built by `python -m recsys.export_web`.

Data: MovieLens 32M. F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems 5, 4. Used for non-commercial research and demonstration under the GroupLens terms; no endorsement by the University of Minnesota or GroupLens is implied.
