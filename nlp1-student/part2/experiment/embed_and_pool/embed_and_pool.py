"""Cross-validate embed-and-pool classifiers on SNLI premise-hypothesis pairs, using the part 1 setup."""

import csv
import inspect
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from multiprocessing import current_process
from pathlib import Path

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from part2.clslib import BasicTokenizer, EmbedAndPoolEncoder
from part2.experiment import evaluate, training_one_epoch
from part2.lib.dataset import DEFAULT_CACHE_DIR, load_snli
from part2.lib.embeddings import MODELS, _load_vectors_generic, download
from project_1.experiment_fork import cross_validation_split

PAD, UNK = 0, 1
RESULTS = Path(__file__).resolve().parent / "results"
ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}

SETTINGS = ["lr", "batch_size", "weight_decay", "epochs", "hidden_size", "num_layers", "dropout", "activation"]
SCORES = ["fold", "best_epoch", "train_acc", "val_loss", "val_acc"]


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def tutorial_head(in_dim, dropout=0.5, hidden_size=50, num_layers=2, activation="tanh"):
    """The tutorial's SoftmaxRegressor layout, dropout before every linear layer."""
    layers, width = [], in_dim
    for _ in range(num_layers):
        layers += [nn.Dropout(dropout), nn.Linear(width, hidden_size), ACTIVATIONS[activation]()]
        width = hidden_size
    layers += [nn.Dropout(dropout), nn.Linear(width, 3)]
    return nn.Sequential(*layers, nn.LogSoftmax(-1))


def token_ids(sentences, model, cache_dir, progress=True):
    """Map each distinct sentence to word ids, and return the matching embedding matrix."""
    tokenizer = BasicTokenizer(lowercase=False, remove_punct=False, remove_stopwords=False)
    distinct = tqdm(dict.fromkeys(sentences), desc="tokenizing", disable=not progress)
    tokens = {s: [t for t in tokenizer.tokens(s) if any(c.isalnum() for c in t)] for s in distinct}
    words = sorted({t for ts in tokens.values() for t in ts})
    keys = {form.encode() for w in words for form in (w, w.lower())}

    log(f"loading {model.name} vectors for {len(words)} words")

    vectors, unknown = _load_vectors_generic(download(model.url, Path(cache_dir) / model.filename), model, keys)
    rows, index = [np.zeros_like(unknown), unknown], {}
    for w in words:
        vector = vectors.get(w.encode(), vectors.get(w.lower().encode()))
        if vector is not None:
            index[w] = len(rows)
            rows.append(vector)
    ids = {s: torch.tensor([index.get(t, UNK) for t in ts], dtype=torch.long) for s, ts in tokens.items()}

    log(f"{len(rows) - 2}/{len(words)} words have a pretrained vector")

    return ids, torch.tensor(np.stack(rows))


class PairEncoder(nn.Module):
    """Pools premise and hypothesis separately, then combines them as [u; v; |u-v|; u*v]."""

    def __init__(self, **kwargs):
        super().__init__()
        self.pool = EmbedAndPoolEncoder(padding_idx=PAD, **kwargs)
        self.enc_dim = 4 * self.pool.enc_dim

    def forward(self, x):  # x: [batch, 2, max_len]
        u, v = self.pool(x).unbind(-2)
        return torch.cat([u, v, (u - v).abs(), u * v], -1)


def loader(rows, ids, batch_size, shuffle):
    data = [(ids[r["premise"]], ids[r["hypothesis"]], r["label"]) for r in rows]

    def collate(batch):
        x = pad_sequence([s for p, h, _ in batch for s in (p, h)], batch_first=True, padding_value=PAD)
        return x.view(len(batch), 2, -1), torch.tensor([y for *_, y in batch])

    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def append_row(path, row, columns):
    """Locked, so parallel runs can append to the same CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", newline="") as f:
        if fcntl:
            fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if not f.readline():
            writer.writeheader()
        writer.writerow(row)


def train(heads, settings, columns, results_file, lr, weight_decay, batch_size, epochs, model="ConceptNet Numberbatch",
          folds=5, seed=42, train_size=None, cache_dir=DEFAULT_CACHE_DIR, progress=True):
    """Cross-validate each head ({name: in_dim -> module}) and append one row per (head, fold) to results_file.
    train_size: first N training pairs only, for quick tests."""
    model_spec = next(m for m in MODELS if m.name == model)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    log("loading SNLI")

    rows = [r for _, r in load_snli("train", cache_dir=cache_dir)][:train_size]
    splits = cross_validation_split(rows, k=folds, seed=seed)
    ids, matrix = token_ids([r[f] for r in rows for f in ("premise", "hypothesis")], model_spec, cache_dir, progress)

    for head, make_head in heads.items():
        for fold, split in splits.items():
            torch.manual_seed(seed)
            encoder = PairEncoder(embedding_matrix=matrix.clone(), average_pool=True, freeze_embeddings=True)
            net = nn.Sequential(encoder, make_head(encoder.enc_dim)).to(device)
            optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
            train_loader = loader(split["training"], ids, batch_size, shuffle=True)
            val_loader = loader(split["validation"], ids, batch_size, shuffle=False)

            log(f"{head} fold {fold + 1}/{len(splits)}: training on {len(split['training'])}, validating on {len(split['validation'])}")

            history = []
            for epoch in range(epochs):
                batches = tqdm(train_loader, desc=f"{head} fold {fold + 1} epoch {epoch + 1}", leave=False, disable=not progress)
                tr = training_one_epoch(net, batches, optimizer, nn.NLLLoss(), device)
                va = evaluate(net, val_loader, nn.NLLLoss(), device)
                history.append({"val_loss": va["loss"], "val_acc": va["accuracy"], "train_acc": tr["accuracy"]})

                log(f"{head} fold {fold + 1} epoch {epoch + 1:02d} | train acc {tr['accuracy']:.4f} | val loss {va['loss']:.4f} | val acc {va['accuracy']:.4f}")

            # like part 1: the epoch with the lowest validation loss, later epochs win ties
            best = min(range(len(history)), key=lambda e: (history[e]["val_loss"], -e))
            record = {**settings, "head": head, "fold": fold + 1, "best_epoch": best + 1, **history[best]}
            append_row(results_file, record, columns)
            
            log(f"{head} fold {fold + 1}: best epoch {best + 1}, val acc {history[best]['val_acc']:.4f}")


def run(dropout=0.5, hidden_size=50, num_layers=2, activation="tanh", lr=1e-3, weight_decay=1e-4, batch_size=256,
        epochs=1, results_file=RESULTS / "scratch.csv", **kwargs):
    """Cross-validate the tutorial head with one configuration; kwargs go to train()."""
    settings = {"lr": lr, "batch_size": batch_size, "weight_decay": weight_decay, "epochs": epochs,
                "hidden_size": hidden_size, "num_layers": num_layers, "dropout": dropout, "activation": activation}
    head = lambda in_dim: tutorial_head(in_dim, dropout, hidden_size, num_layers, activation)
    train({"tutorial": head}, settings, SETTINGS + SCORES, results_file, lr, weight_decay, batch_size, epochs, **kwargs)


SHORT_NAMES = {"hidden_size": "w", "num_layers": "l", "lr": "lr", "batch_size": "bs", "activation": "", "dropout": "dropout"}


def config_name(config):  # e.g. {"hidden_size": 60, "num_layers": 3} -> w60-l3
    return "-".join(f"{SHORT_NAMES[k]}{v:g}" if isinstance(v, float | int) else f"{SHORT_NAMES[k]}{v}" for k, v in config.items())


def _run_quiet(job):
    config, settings, results_file = job
    with open(os.devnull, "w") as f, redirect_stdout(f), redirect_stderr(f):
        run(**config, **settings, results_file=results_file, progress=False)
    return config_name(config)


def sweep(name, configs, max_workers=4, **settings):
    """Run each config in its own process, appending to results/<name>.csv.
    Configs already in the CSV with all folds are skipped, so an interrupted sweep can just be rerun."""
    if current_process().name != "MainProcess":  # workers re-run the sweep script on start; only the main one sweeps
        return
    results_file = RESULTS / f"{name}.csv"

    defaults = {k: p.default for k, p in inspect.signature(run).parameters.items()}
    key = lambda values: tuple(str(values[k]) for k in SETTINGS)
    rows = list(csv.DictReader(results_file.open())) if results_file.exists() else []
    counts = Counter(map(key, rows))
    done = {k for k, n in counts.items() if n >= settings.get("folds", 5)}
    if len(done) < len(counts):  # drop rows of half-finished configs, they are rerun
        with results_file.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SETTINGS + SCORES)
            writer.writeheader()
            writer.writerows(r for r in rows if key(r) in done)
    todo = [c for c in configs if key({**defaults, **settings, **c}) not in done]
    print(f"{len(configs) - len(todo)} configs already done, running {len(todo)}")

    jobs = [(c, settings, results_file) for c in todo]
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for finished in pool.map(_run_quiet, jobs):
            print(f"{finished} done", flush=True)


if __name__ == "__main__":
    run(train_size=3000)  # quick smoke test
