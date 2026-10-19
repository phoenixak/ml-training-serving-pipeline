"""FastAPI service: recommendations, similar movies and title search, served with onnxruntime.

uvicorn recsys.api:app --app-dir src --port 8000      (ARTIFACTS=artifacts by default)
"""

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from recsys.serve import Recommender


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not getattr(app.state, "rec", None):
        app.state.rec = Recommender(
            os.environ.get("ARTIFACTS", "artifacts"), threads=int(os.environ.get("THREADS", "2"))
        )
    yield


app = FastAPI(title="Two-stage movie recommender", version="2.0.0", lifespan=lifespan)


class RecommendRequest(BaseModel):
    liked: list[int] = Field(
        min_length=1, max_length=200, description="MovieLens movieIds the user liked, oldest first"
    )
    k: int = Field(10, ge=1, le=50)
    rerank: bool = True


@app.middleware("http")
async def timing(request: Request, call_next):
    t = time.perf_counter()
    response = await call_next(request)
    response.headers["Server-Timing"] = f"app;dur={1000 * (time.perf_counter() - t):.1f}"
    return response


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "items": len(request.app.state.rec.items)}


@app.get("/movies/search")
def search(request: Request, q: str = Query(min_length=2), limit: int = Query(20, le=50)) -> list[dict]:
    return request.app.state.rec.search(q, limit)


@app.post("/recommend")
def recommend(request: Request, body: RecommendRequest) -> dict:
    rec: Recommender = request.app.state.rec
    known = rec.to_idx(body.liked)
    if not known:
        raise HTTPException(422, "None of the liked movieIds are in the catalogue")
    return {"items": rec.recommend(body.liked, body.k, rerank=body.rerank), "used_history": len(known)}


@app.get("/similar/{movie_id}")
def similar(request: Request, movie_id: int, k: int = Query(10, le=50)) -> list[dict]:
    out = request.app.state.rec.similar(movie_id, k)
    if not out:
        raise HTTPException(404, "Unknown movieId")
    return out
