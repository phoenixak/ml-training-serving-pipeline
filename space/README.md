---
title: Movie Recommender
emoji: 🎬
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: other
short_description: Two-tower retrieval + DLRM ranking on MovieLens-32M
---

# Movies you'll like next

Pick a few movies you enjoyed and get ten recommendations, each with the movie of yours it is closest to. A two-tower model trained on MovieLens-32M turns your picks into a taste vector and retrieves 100 candidates from the catalogue; a DLRM-style ranker re-orders them. Untick re-ranking to compare. Everything runs with ONNX Runtime on CPU in milliseconds, and nothing about you is stored.

Code, training and evaluation: [github.com/phoenixak/ml-training-serving-pipeline](https://github.com/phoenixak/ml-training-serving-pipeline)

**Data:** MovieLens 32M — F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems 5, 4. The model and the movie list derived from it are shared for non-commercial research and demonstration under the GroupLens usage terms; this Space is not endorsed by the University of Minnesota or GroupLens. Movie links go to The Movie Database.
