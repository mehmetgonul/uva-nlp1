"""Load the SNLI training pairs and their frozen ConceptNet Numberbatch vectors."""

import hashlib
import json
import random
import re
from dataclasses import dataclass
from functools import lru_cache
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Subset

from part2.clslib import BasicTokenizer, PretrainedEmbeddings, Vocabulary
from part2.lib.dataset import DEFAULT_CACHE_DIR, download
from part2.lib.embeddings import CONCEPTNET

from .experiment import LabelledCorpus

SNLI_LABELS = ("entailment", "neutral", "contradiction")


@lru_cache(maxsize=65536)
def numberbatch_term(word):
    """Numberbatch masks digits in words that contain two consecutive digits."""
    word = word.lower()
    return re.sub(r"\d", "#", word) if re.search(r"\d\d", word) else word


@dataclass
class NumberbatchTable:
    vocab: Vocabulary
    vectors: np.ndarray
    source: str
    pad_token: str = "<pad>"


def load_numberbatch():
    path = download(CONCEPTNET.url, DEFAULT_CACHE_DIR / CONCEPTNET.filename)
    embeddings = PretrainedEmbeddings(str(path), num_dims=300, skip_first_line=True)
    vocab = embeddings.vocab
    vectors = embeddings.vectors
    if "<sep>" not in vocab:
        # Append the SNLI separator after the tutorial loader has computed <unk>.
        vocab = Vocabulary([*vocab, "<sep>"], default=embeddings.unk_token, frozen=True)
        separator = np.random.default_rng(42).normal(0, 0.02, size=embeddings.num_dims)
        vectors = np.vstack((vectors, separator.astype(vectors.dtype)))
    return NumberbatchTable(vocab, vectors, CONCEPTNET.url)


def load_snli_training(cache_dir):
    """Load labelled SNLI training pairs as premise IDs + <sep> + hypothesis IDs."""
    data = load_dataset("stanfordnlp/snli", split="train", cache_dir=str(cache_dir)).to_pandas()
    data = data.loc[data["label"].isin([0, 1, 2])].dropna(subset=["premise", "hypothesis"])

    tokenizer = BasicTokenizer(
        lowercase=True,
        remove_punct=False,
        remove_stopwords=False,
        language="english",
    )
    table = load_numberbatch()
    vocab = table.vocab
    sentences = {}
    for text in chain(data["premise"], data["hypothesis"]):
        if text not in sentences:
            tokens = tokenizer.tokens(text)
            sentences[text] = [vocab.index(numberbatch_term(token)) for token in tokens]

    sep_id = vocab.index("<sep>")
    inputs = (
        sentences[premise] + [sep_id] + sentences[hypothesis]
        for premise, hypothesis in zip(data["premise"], data["hypothesis"])
    )
    corpus = LabelledCorpus(inputs, data["label"].to_numpy(dtype=np.int64))
    return corpus, table, data.index.to_numpy()


def save_numberbatch(table, output_dir):
    """Save the frozen vectors once so each fold only needs its learned weights."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(json.dumps(table.vocab.words).encode())
    digest.update(table.vectors.data)
    path = output_dir / f"numberbatch-{digest.hexdigest()[:16]}.pt"
    if not path.is_file():
        with TemporaryDirectory(dir=output_dir) as temporary:
            staged = Path(temporary) / path.name
            torch.save(
                {
                    "vectors": torch.from_numpy(table.vectors),
                    "words": table.vocab.words,
                    "source": table.source,
                    "labels": SNLI_LABELS,
                    "pad_id": table.vocab.index("<pad>"),
                    "unk_id": table.vocab.default_index,
                    "sep_id": table.vocab.index("<sep>"),
                },
                staged,
            )
            staged.replace(path)
    return path


def cross_validation_split(corpus, row_ids, num_folds, seed):
    """Shuffle and partition the training pairs in the same order as Project 1."""
    if not 2 <= num_folds <= len(corpus):
        raise ValueError("num_folds must be between 2 and the number of labelled pairs")
    positions = list(range(len(corpus)))
    random.Random(seed).shuffle(positions)
    parts = np.array_split(positions, num_folds)

    folds = {}
    for fold, validation in enumerate(parts):
        training = np.concatenate(parts[:fold] + parts[fold + 1:])
        folds[fold] = {
            "training": Subset(corpus, training),
            "validation": Subset(corpus, validation),
            "snli_rows": row_ids[validation],
        }
    return folds
