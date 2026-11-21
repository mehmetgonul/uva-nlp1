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

from .clslib import BasicTokenizer, Vocabulary
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


def load_numberbatch(sentences, cache_dir):
    cache_file = Path(cache_dir) / "snli_embedding_similarity" / "numberbatch.npz"
    if cache_file.is_file():
        with np.load(cache_file, allow_pickle=False) as cache:
            pairs = zip(cache["words"].tolist(), cache["vectors"], strict=True)
            word_vectors = {numberbatch_term(word): vec for word, vec in pairs if vec.any()}
        source = cache_file.name
    else:
        # The project's checkpoint reader also recovers missing terms by matching prefixes.
        from part2.lib.dataset import DEFAULT_CACHE_DIR, download
        from part2.lib.embeddings import CONCEPTNET, _load_vectors_conceptnet

        path = download(CONCEPTNET.url, DEFAULT_CACHE_DIR / CONCEPTNET.filename)
        terms = {term.encode() for tokens in sentences for term in tokens}
        matches, _, _ = _load_vectors_conceptnet(path, terms)
        word_vectors = {}
        for key, vec in matches.items():
            norm = np.linalg.norm(vec)
            if norm > 0:
                word_vectors[key.decode()] = vec / norm
        source = CONCEPTNET.url

    words = ["<pad>", "<unk>", "<sep>", *sorted(word_vectors)]
    vocab = Vocabulary(words, default="<unk>", frozen=True)
    vectors = np.zeros((len(vocab), 300), dtype=np.float32)
    vectors[vocab.index("<sep>")] = np.random.default_rng(42).normal(0, 0.02, size=300)
    for word, vector in word_vectors.items():
        vectors[vocab.index(word)] = vector
    return NumberbatchTable(vocab, vectors, source)


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
    sentences = {}
    for text in chain(data["premise"], data["hypothesis"]):
        if text not in sentences:
            tokens = tokenizer.tokens(text)
            sentences[text] = [numberbatch_term(token) for token in tokens]

    table = load_numberbatch(sentences.values(), cache_dir)
    vocab = table.vocab
    for text, tokens in sentences.items():
        sentences[text] = [vocab.index(token) for token in tokens]

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
