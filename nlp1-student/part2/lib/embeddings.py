"""Load pretrained word vectors for the tokens in each sentence."""

import gzip
import re
from dataclasses import dataclass
from functools import lru_cache
from io import BufferedReader
from pathlib import Path

import numpy as np

from ..clslib import BasicTokenizer, PretrainedEmbeddings, Vocabulary
from .dataset import DEFAULT_CACHE_DIR, FIELDS, download


@dataclass(frozen=True)
class Model:
    name: str
    filename: str
    url: str
    binary: bool = False


def gensim_model(name, filename, release, binary=False):
    url = f"https://github.com/piskvorky/gensim-data/releases/download/{release}/{release}.gz"
    return Model(name, filename, url, binary)


CONCEPTNET = Model(
    "ConceptNet Numberbatch",
    "numberbatch.gz",
    "https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz",
)

GLOVE = gensim_model("GloVe", "glove.gz", "glove-wiki-gigaword-300")
FASTTEXT = gensim_model("fastText", "fasttext.gz", "fasttext-wiki-news-subwords-300")

MODELS = (
    GLOVE,
    FASTTEXT,
    CONCEPTNET,
    gensim_model("Google News Word2Vec", "word2vec.gz", "word2vec-google-news-300", binary=True),
)


def snli_tokenizer():
    return BasicTokenizer(lowercase=True, remove_punct=False, remove_stopwords=False, language="english")


@lru_cache(maxsize=65536)
def numberbatch_term(word):
    """Numberbatch masks digits in words that contain two consecutive digits."""
    word = word.lower()
    return re.sub(r"\d", "#", word) if re.search(r"\d\d", word) else word


def load_numberbatch_embeddings(cache_dir):
    path = download(CONCEPTNET.url, Path(cache_dir) / CONCEPTNET.filename)
    return PretrainedEmbeddings(str(path), num_dims=300, skip_first_line=True)


def _binary_records(file, count, dimensions):
    vector_size = dimensions * np.dtype("<f4").itemsize
    for _ in range(count):
        # Google News stores each word up to a space, then its float32 vector.
        word = bytearray()
        byte = file.read(1)
        while byte and byte != b" ":
            word.extend(byte)
            byte = file.read(1)
        vector = file.read(vector_size)
        if len(vector) != vector_size:
            raise EOFError(f"Truncated word2vec record for {bytes(word)!r}")
        yield bytes(word).lstrip(b"\n"), vector
    file.read()  # Consume the trailing newline and validate the gzip footer.


def _text_vector(path, word, data, dimensions):
    vector = np.fromstring(data, sep=" ", dtype=np.float32)
    if vector.size != dimensions:
        raise ValueError(f"{path}: invalid vector length for {word!r}")
    return vector


def _load_vectors_generic(path, model, vocabulary):
    # ponytail: these checkpoints have a word2vec header and single-token keys.
    # Revisit the reader if the experiment adds a different model format.
    vectors = {}
    with gzip.open(path, "rb") as compressed, BufferedReader(compressed) as file:
        count, dimensions = map(int, file.readline().split())
        vector_sum = np.zeros(dimensions, dtype=np.float64)
        vector_count = 0
        if model.binary:
            for word, data in _binary_records(file, count, dimensions):
                vector = np.frombuffer(data, dtype="<f4")
                vector_sum += vector
                if word in vocabulary:
                    vectors.setdefault(word, vector)
            vector_count = count
        else:
            for line in file:
                word, _, data = line.partition(b" ")
                vector = _text_vector(path, word, data, dimensions)
                vector_sum += vector
                vector_count += 1
                if word in vocabulary:
                    vectors.setdefault(word, vector)
    if vector_count == 0:
        raise ValueError(f"{path}: no vectors for the unknown-token mean")
    unknown_vector = (vector_sum / vector_count).astype(np.float32)
    return vectors, unknown_vector


def load_word_vectors(model, words, *, cache_dir=DEFAULT_CACHE_DIR):
    """Load exact vocabulary matches and the model's mean vector for unknown words."""
    path = download(model.url, Path(cache_dir) / model.filename)
    keys = {word.encode() for word in words}
    vectors, unknown = _load_vectors_generic(path, model, keys)
    return {key.decode(): vector for key, vector in vectors.items()}, unknown


@dataclass(frozen=True)
class SentenceEmbedding:
    values: np.ndarray
    token_count: int
    matched_tokens: int


def embed_sentences(sentences, model, *, cache_dir=DEFAULT_CACHE_DIR):
    """Return word vectors per distinct sentence, keyed by its original text."""
    tokenizer = snli_tokenizer()
    tokens_by_sentence = {}
    for sentence in dict.fromkeys(sentences):
        tokens_by_sentence[sentence] = [
            token for token in tokenizer.tokens(sentence)
            if any(character.isalnum() for character in token)
        ]
    words = (token for tokens in tokens_by_sentence.values() for token in tokens)
    vectors, default_row = load_word_vectors(model, words, cache_dir=cache_dir)
    vocab = Vocabulary(["", *vectors], default="", frozen=True)
    matrix = np.stack([default_row, *vectors.values()])

    embeddings = {}
    for sentence, tokens in tokens_by_sentence.items():
        ids = [vocab.index(token) for token in tokens]
        matched_tokens = sum(index != vocab.default_index for index in ids)
        embeddings[sentence] = SentenceEmbedding(matrix[ids], len(ids), matched_tokens)
    return embeddings


def embed_snli_pairs(pairs, model, *, cache_dir=DEFAULT_CACHE_DIR):
    """Yield (row index, gold label ID, premise embedding, hypothesis embedding)."""
    pairs = list(pairs)
    sentences = []
    for _, pair in pairs:
        for field in FIELDS:
            sentences.append(pair[field])

    embeddings = embed_sentences(sentences, model, cache_dir=cache_dir)
    for row_id, pair in pairs:
        yield row_id, pair["label"], embeddings[pair["premise"]], embeddings[pair["hypothesis"]]
