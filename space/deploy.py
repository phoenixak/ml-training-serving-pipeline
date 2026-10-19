"""Copy the Space (app, serving code and exported artifacts) into a folder, ready for `hf upload`.

python space/deploy.py ../hf-movie-recommender
hf upload Phoenixak99/movie-recommender ../hf-movie-recommender . --repo-type space --delete "*"
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = ["app.py", "style.css", "requirements.txt", "README.md", "Dockerfile"]
ARTIFACTS = [
    "user_tower.onnx",
    "ranker.onnx",
    "item_vectors.npy",
    "item_genres.npy",
    "item_decades.npy",
    "item_log_pop.npy",
    "items.json",
    "manifest.json",
    "metrics.json",
]


def main(target: str) -> None:
    dst = Path(target).resolve()
    dst.mkdir(parents=True, exist_ok=True)
    for item in dst.iterdir():
        if item.name not in {".git", ".gitattributes"}:
            shutil.rmtree(item) if item.is_dir() else item.unlink()
    for name in FILES:
        shutil.copy2(ROOT / "space" / name, dst / name)
    (dst / "src" / "recsys").mkdir(parents=True)
    for name in ("__init__.py", "serve.py"):  # serving only needs these
        shutil.copy2(ROOT / "src" / "recsys" / name, dst / "src" / "recsys" / name)
    (dst / "artifacts").mkdir()
    for name in ARTIFACTS:
        shutil.copy2(ROOT / "artifacts" / name, dst / "artifacts" / name)
    print(f"copied the Space into {dst}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT.parent / "hf-movie-recommender"))
