"""Export what serving needs: item vectors, item metadata, and ONNX graphs for the user tower and ranker.

Serving never imports PyTorch: onnxruntime runs both graphs and numpy does the retrieval.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from recsys.data import Data
from recsys.model import MAX_HISTORY, DLRMRanker, TwoTower


class _UserGraph(nn.Module):
    def __init__(self, model: TwoTower) -> None:
        super().__init__()
        self.model = model

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.model.user_vectors(history)


def export(model: TwoTower, ranker: DLRMRanker, data: Data, out: Path) -> None:
    model.eval()
    ranker.eval()
    with torch.no_grad():
        items = model.items.all_items().numpy().astype(np.float32)
    np.save(out / "item_vectors.npy", items)
    np.save(out / "item_genres.npy", data.genre_matrix())
    np.save(out / "item_decades.npy", data.decade_index())
    np.save(out / "item_log_pop.npy", np.log(data.popularity()).astype(np.float32))
    meta = data.items.assign(year=data.items["year"].fillna(0).astype(int))
    meta[["idx", "movieId", "tmdbId", "title", "year", "genres", "n_likes"]].to_json(
        out / "items.json", orient="records"
    )

    history = torch.zeros(1, MAX_HISTORY, dtype=torch.long)
    history[0, :3] = torch.tensor([1, 2, 3])
    # wrappers must be in eval mode too: the exporter restores the wrapper's mode afterwards, which
    # would otherwise switch the shared model back to training (dropout on)
    torch.onnx.export(
        _UserGraph(model).eval(),
        (history,),
        out / "user_tower.onnx",
        input_names=["history"],
        output_names=["user_vec"],
        dynamic_axes={"history": {0: "batch"}, "user_vec": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    n = 4
    example = {
        "item_idx": torch.arange(1, n + 1),
        "item_genres": torch.zeros(n, ranker.genre.in_features),
        "item_decade": torch.zeros(n, dtype=torch.long),
        "user_vec": torch.zeros(n, ranker.user_proj.in_features),
        "item_vec": torch.zeros(n, ranker.item_proj.in_features),
        "dense": torch.zeros(n, DLRMRanker.N_DENSE),
    }
    torch.onnx.export(
        ranker,
        tuple(example.values()),
        out / "ranker.onnx",
        input_names=list(example),
        output_names=["score"],
        dynamic_axes={k: {0: "n"} for k in [*example, "score"]},
        opset_version=17,
        dynamo=False,
    )
    model.eval()
    ranker.eval()
    (out / "manifest.json").write_text(
        json.dumps({"max_history": MAX_HISTORY, "n_items": int(items.shape[0]), "dim": int(items.shape[1])}, indent=2)
    )
    print(f"exported to {out}: user_tower.onnx, ranker.onnx, item_vectors.npy ({items.shape}), items.json")
