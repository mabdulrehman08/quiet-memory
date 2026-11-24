"""Fetch pinned public benchmark and model files into the ignored cache."""

from pathlib import Path
from urllib.request import urlretrieve

CACHE = Path(__file__).resolve().parent / ".cache"
LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
MODEL_URL = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def fetch():
    CACHE.mkdir(exist_ok=True)
    files = {
        "locomo10.json": f"{LOCOMO_URL}/data/locomo10.json",
        "tokenizer.json": f"{MODEL_URL}/tokenizer.json",
        "model.onnx": f"{MODEL_URL}/onnx/model.onnx",
    }
    for name, url in files.items():
        destination = CACHE / name
        if not destination.exists():
            print(f"Fetching {name}", flush=True)
            temporary = destination.with_suffix(".part")
            urlretrieve(url, temporary)
            temporary.replace(destination)


if __name__ == "__main__":
    fetch()
