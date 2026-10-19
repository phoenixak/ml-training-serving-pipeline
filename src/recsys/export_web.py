"""Package the exported artifacts for the in-browser demo (onnxruntime-web, no server).

    python -m recsys.export_web --artifacts artifacts --out web/model

Arrays become little-endian binaries the page reads with typed arrays; shapes go in meta.json.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="web/model")
    args = ap.parse_args()
    a, out = Path(args.artifacts), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    vectors = np.load(a / "item_vectors.npy").astype("<f4")
    genres = np.load(a / "item_genres.npy").astype("<f4")
    decades = np.load(a / "item_decades.npy").astype("<i4")
    log_pop = np.load(a / "item_log_pop.npy").astype("<f4")
    for name, arr in [("vectors", vectors), ("genres", genres), ("decades", decades), ("log_pop", log_pop)]:
        (out / f"{name}.bin").write_bytes(arr.tobytes())
    for f in ("user_tower.onnx", "ranker.onnx"):
        shutil.copy2(a / f, out / f)

    items = json.loads((a / "items.json").read_text(encoding="utf-8"))
    slim = [[m["idx"], m["movieId"], m["title"], m["year"], [g for g in m["genres"] if g != "(no genres listed)"],
             int(m["tmdbId"]) if m.get("tmdbId") == m.get("tmdbId") and m.get("tmdbId") else None, m["n_likes"]]
            for m in items]
    (out / "items.json").write_text(json.dumps(slim, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    manifest = json.loads((a / "manifest.json").read_text())
    meta = {**manifest, "n_genres": int(genres.shape[1]), "fields": ["idx", "movieId", "title", "year", "genres", "tmdbId", "n_likes"]}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    total = sum(p.stat().st_size for p in out.iterdir())
    print(f"wrote {out} ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
