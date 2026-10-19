"""Hugging Face Space: pick a few movies you like, get recommendations from a two-stage recommender."""

import html
import os
import sys
import time
from pathlib import Path

import gradio as gr

HERE = Path(__file__).resolve().parent
for candidate in (HERE / "src", HERE.parent / "src"):
    if candidate.exists():
        sys.path.insert(0, str(candidate))

from recsys.serve import Recommender  # noqa: E402

ARTIFACTS = next(p for p in (HERE / "artifacts", HERE.parent / "artifacts") if p.exists())
rec = Recommender(ARTIFACTS, threads=int(os.environ.get("THREADS", "2")))

ORDERED = sorted(rec.items.values(), key=lambda m: -m["n_likes"])
LABEL = {m["movieId"]: f"{m['title']} ({m['year']})" if m["year"] else m["title"] for m in ORDERED}
BY_LABEL = {v: k for k, v in LABEL.items()}
CHOICES = [LABEL[m["movieId"]] for m in ORDERED]


def preset(*titles: str) -> list[str]:
    out = []
    for t in titles:
        hit = rec.search(t, 1)
        if hit:
            out.append(LABEL[hit[0]["movieId"]])
    return out


PRESETS = {
    "90s sci-fi fan": preset("Matrix", "Terminator 2", "Jurassic Park", "Twelve Monkeys", "Fifth Element"),
    "Animation lover": preset("Toy Story", "Spirited Away", "Up", "WALL·E", "Inside Out"),
    "Crime & thrillers": preset("Pulp Fiction", "Se7en", "Usual Suspects", "Departed", "Prisoners"),
    "Romantic comedies": preset(
        "When Harry Met Sally", "Notting Hill", "Love Actually", "Crazy, Stupid, Love", "About Time"
    ),
}


def card(m: dict) -> str:
    genres = " · ".join(g for g in m["genres"] if g != "(no genres listed)")
    link = (
        f"https://www.themoviedb.org/movie/{int(m['tmdbId'])}"
        if m.get("tmdbId") == m.get("tmdbId") and m.get("tmdbId")
        else None
    )
    title = html.escape(m["title"])
    title = f"<a href='{link}' target='_blank' rel='noopener'>{title} ↗</a>" if link else title
    return (
        f"<div class='card'><p class='yr'>{m['year'] or ''}</p><h4>{title}</h4><p class='g'>{html.escape(genres)}</p>"
        f"<p class='why'>because you liked <b>{html.escape(m['because'])}</b></p></div>"
    )


def recommend(picked: list[str], rerank: bool):
    liked = [BY_LABEL[p] for p in picked or [] if p in BY_LABEL]
    if not liked:
        return "<p class='muted'>Pick at least one movie you like (or use a preset).</p>", ""
    t = time.perf_counter()
    items = rec.recommend(liked, k=10, rerank=rerank)
    ms = 1000 * (time.perf_counter() - t)
    how = "two-tower retrieval → DLRM re-ranking of the top 100" if rerank else "two-tower retrieval only"
    return "<div class='grid'>" + "".join(card(m) for m in items) + "</div>", f"{how} · {ms:.0f} ms on CPU"


CSS = (HERE / "style.css").read_text(encoding="utf-8")

with gr.Blocks(title="Movie recommender — two-stage") as demo:
    gr.HTML(
        "<header class='hd'><p class='kicker'>Two-tower retrieval · DLRM ranking · ONNX on CPU</p>"
        "<h1>Movies you'll like next</h1><p class='lede'>Pick a few movies you enjoyed. A two-tower model trained on "
        "MovieLens-32M turns them into a taste vector and pulls 100 candidates out of 4,779 movies; a DLRM-style "
        "ranker re-orders them. Nothing about you is stored — the model works from the movies you pick.</p>"
        "<p class='links'><a href='https://github.com/phoenixak/ml-training-serving-pipeline' target='_blank'>Code & evaluation ↗</a>"
        "<a href='https://ahmed.fynxforge.com' target='_blank'>Ahmed Khaled Ali ↗</a></p></header>"
    )
    with gr.Row(equal_height=False):
        with gr.Column(scale=4):
            picked = gr.Dropdown(
                CHOICES,
                multiselect=True,
                filterable=True,
                label="Movies you like",
                value=PRESETS["90s sci-fi fan"],
                max_choices=30,
            )
            with gr.Row():
                buttons = [gr.Button(name, size="sm") for name in PRESETS]
            rerank = gr.Checkbox(True, label="Re-rank with the DLRM ranker (untick to compare)")
            go = gr.Button("Recommend", variant="primary")
            stats = gr.Markdown()
        with gr.Column(scale=7):
            out = gr.HTML("<p class='muted'>Recommendations appear here.</p>")
    gr.HTML(
        "<footer class='ft'>Data: MovieLens 32M — F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens "
        "Datasets: History and Context. ACM TiiS 5, 4. Used for non-commercial research and demonstration under "
        "the GroupLens terms; no endorsement by the University of Minnesota or GroupLens is implied. "
        "Links go to The Movie Database.</footer>"
    )

    for b, name in zip(buttons, PRESETS):
        b.click(lambda n=name: PRESETS[n], None, picked).then(recommend, [picked, rerank], [out, stats])
    go.click(recommend, [picked, rerank], [out, stats])
    rerank.change(recommend, [picked, rerank], [out, stats])
    demo.load(recommend, [picked, rerank], [out, stats])

if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Base())
