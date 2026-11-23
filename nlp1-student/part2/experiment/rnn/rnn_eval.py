"""Evaluate saved ConceptNet LSTM folds on the embed-and-pool SNLI error-analysis subsets."""

import json
import statistics
import sys
from functools import cache, partial
from itertools import chain, product
from pathlib import Path


SEEDS = [42, 43, 44]
NUM_FOLDS = 5
LSTM_HIDDEN_SIZES = [40]
LSTM_NUM_LAYERS = [4]
LSTM_BIDIRECTIONAL = [False]
CLASSIFIER_HIDDEN_SIZES = [(50, 50)]
CLASSIFIER_DROPOUT_RATES = [0.1]
LEARNING_RATES = [3e-3]
WEIGHT_DECAYS = [1e-4]
BATCH_SIZES = [128]
NUM_EPOCHS = [20]

OUTPUT_DIR = Path("/Users/mehmetvecdigonul/git/uva-nlp1/nlp1-student/part2/experiment/rnn/output/conceptnet")
EVAL_BATCH_SIZE = 512

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "embed_and_pool"), str(HERE.parent), str(HERE.parents[2])]

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from error_analysis import length, overlap, subsets
from part2.lib.dataset import DEFAULT_CACHE_DIR, load_snli
from rnn import data, experiment
from rnn.rnn_main import EMBEDDING_MODELS, lstm_run_stem


def selected_runs():
    grid = product(
        EMBEDDING_MODELS, LSTM_HIDDEN_SIZES, LSTM_NUM_LAYERS, LSTM_BIDIRECTIONAL,
        CLASSIFIER_HIDDEN_SIZES, CLASSIFIER_DROPOUT_RATES,
        LEARNING_RATES, WEIGHT_DECAYS, BATCH_SIZES, NUM_EPOCHS,
    )
    for (
        embedding_model, hidden_size, depth, bidirectional, widths, dropout,
        lr, weight_decay, batch_size, epochs,
    ) in grid:
        head = "-".join(map(str, widths)) or "linear"
        stem = lstm_run_stem(depth, hidden_size, bidirectional, embedding_model)
        name = (
            f"{stem}-mlp{head}-drop{dropout}"
            f"-lr{lr}-wd{weight_decay}-batch{batch_size}-epochs{epochs}-cv{NUM_FOLDS}"
        )
        for seed in SEEDS:
            run_dir = OUTPUT_DIR / name / f"seed-{seed}"
            models = []
            for fold in range(NUM_FOLDS):
                folder = run_dir / f"fold-{fold}"
                if all((folder / file).is_file() for file in ("metric.json", "model.json", "model.pt")):
                    models.append(folder / "model")
            print(f"\n{name}, seed {seed} ({len(models)}/{NUM_FOLDS} completed folds)", flush=True)
            if models:
                yield run_dir, models


def make_loader(rows, table_path):
    """Encode each pair as premise IDs + <sep> + hypothesis IDs, as in rnn/data.py."""
    table = torch.load(table_path, map_location="cpu", weights_only=True, mmap=True)
    words = table["words"]
    vocab = data.Vocabulary(words, default=words[table["unk_id"]], frozen=True)
    tokenizer = data.BasicTokenizer(lowercase=True, remove_punct=False, remove_stopwords=False)
    sentences = {}
    for row in rows:
        for text in (row["premise"], row["hypothesis"]):
            if text not in sentences:
                tokens = tokenizer.tokens(text)
                sentences[text] = [vocab.index(data.numberbatch_term(token)) for token in tokens]
    inputs = (sentences[r["premise"]] + [table["sep_id"]] + sentences[r["hypothesis"]] for r in rows)
    corpus = experiment.LabelledCorpus(inputs, [r["label"] for r in rows])
    return DataLoader(corpus, batch_size=EVAL_BATCH_SIZE, shuffle=False,
                      collate_fn=partial(experiment.collate_labelled_texts, padding_value=table["pad_id"]))


def analysis_groups(rows):
    # Keep row positions so every subset uses the same predictions for a fold.
    indexed = [dict(row, position=i) for i, row in enumerate(rows)]
    groups = [("Validation + test (all)", "full dataset", torch.arange(len(rows)))]
    for name, subset in subsets(indexed).items():
        if "overlap" in name:
            values = [overlap(r) for r in subset]
            span = f"overlap {min(values):.2f}-{max(values):.2f}"
        else:
            values = [length(r) for r in subset]
            span = f"{min(values)}-{max(values)} tokens"
        positions = torch.tensor([r["position"] for r in subset])
        groups.append((name[0].upper() + name[1:], span, positions))
    return groups


def table(models, groups, loader_for, device):
    accs = {name: [] for name, _, _ in groups}
    for path in models:
        with path.with_suffix(".json").open(encoding="utf-8") as f:
            table_path = path.parent / json.load(f)["embedding_table"]
        loader = loader_for(table_path.resolve())
        net = experiment.load(path, device=device)
        metrics = experiment.evaluate(net, loader, torch.nn.NLLLoss(), device)
        predictions = torch.tensor(list(chain.from_iterable(metrics["predictions"])))
        correct = predictions == loader.dataset.y
        for name, _, positions in groups:
            accs[name].append(100 * correct[positions].float().mean().item())
    return [(name, span, len(positions), statistics.mean(accs[name]),
             statistics.stdev(accs[name]) if len(models) > 1 else None)
            for name, span, positions in groups]


def format_accuracy(acc, std):
    return f"{acc:.2f} ± {std:.2f}" if std is not None else f"{acc:.2f} (SD n/a)"


def save_table(rows, path, title):
    columns = ["", "Pairs", "Accuracy (%)", "Range"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.axis("off")
    table = ax.table(cellText=[[name, n, format_accuracy(acc, std), span] for name, span, n, acc, std in rows],
                     colLabels=columns, colWidths=[0.34, 0.16, 0.22, 0.26], cellLoc="center", loc="center")
    table.scale(1, 2)
    table.set_fontsize(11)
    for col in range(len(columns)):
        table[1, col].set_facecolor("#dbe8fb")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {path}")


def main(cache_dir=DEFAULT_CACHE_DIR):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    rows = None

    @cache
    def loader_for(table_path):
        return make_loader(rows, table_path)

    for run_dir, models in selected_runs():
        if rows is None:
            rows = [r for split in ("validation", "test") for _, r in load_snli(split, cache_dir=cache_dir)]
            groups = analysis_groups(rows)
        results = table(models, groups, loader_for, device)
        print(f"{'':<26}{'Pairs':>8}{'Accuracy (%)':>17}   Range")
        for name, span, n, acc, std in results:
            print(f"{name:<26}{n:>8}{format_accuracy(acc, std):>17}   {span}")
        architecture = run_dir.parent.name.split("-mlp")[0]
        title = f"{architecture}, {run_dir.name}, mean ± std over {len(models)} folds"
        save_table(results, run_dir / "error_analysis.png", title)


if __name__ == "__main__":
    main()
