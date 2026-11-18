"""Download shared inputs and load labelled SNLI pairs."""

import shutil
import ssl
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

import certifi
from pyarrow.parquet import read_table

DEFAULT_CACHE_DIR = Path.home() / ".cache/snli-embeddings"
SNLI_REVISION = "cdb5c3d5eed6ead6e5a341c8e56e669bb666725b"
SNLI_SPLITS = ("train", "validation", "test")
FIELDS = ("premise", "hypothesis")
LABELS = ("entailment", "neutral", "contradiction")


def download(url, path):
    if path.is_file():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context(cafile=certifi.where())
    with TemporaryDirectory(dir=path.parent) as directory:
        temporary_path = Path(directory) / path.name
        with urlopen(url, context=context, timeout=60) as response:
            with temporary_path.open("wb") as temporary:
                shutil.copyfileobj(response, temporary)
        temporary_path.replace(path)
    return path


def load_snli(split="test", *, cache_dir=DEFAULT_CACHE_DIR):
    """Return (original row index, row) pairs with a gold label."""
    if split not in SNLI_SPLITS:
        raise ValueError(f"Unknown SNLI split {split!r}. Expected one of {SNLI_SPLITS}.")
    url = f"https://huggingface.co/datasets/stanfordnlp/snli/resolve/{SNLI_REVISION}/plain_text/{split}-00000-of-00001.parquet"
    data_path = download(url, Path(cache_dir) / f"snli-{SNLI_REVISION[:8]}-{split}.parquet")
    rows = read_table(data_path, columns=[*FIELDS, "label"]).to_pylist()
    pairs = []
    for row_id, row in enumerate(rows):
        if row["label"] >= 0:
            pairs.append((row_id, row))
    return pairs
