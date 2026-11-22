import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

import matplotlib.pyplot as plt
import torch
from torch.nn.utils.rnn import pad_sequence

from embed_and_pool import PAD, RESULTS, encode, load_fold, tokenize
from part2.lib.dataset import DEFAULT_CACHE_DIR, load_snli

MODELS = sorted((RESULTS / "best_cv_30ep" / "tutorial").glob("fold-*.pt"))


def predict(net, index, rows, batch_size=512):
    """Predicted label ids for a list of SNLI rows."""
    preds = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            sentences = [encode(r[f], index) for r in batch for f in ("premise", "hypothesis")]
            x = pad_sequence(sentences, batch_first=True, padding_value=PAD).view(len(batch), 2, -1)
            preds.append(net(x).argmax(-1))
    return torch.cat(preds)


def length(row):
    """Premise plus hypothesis tokens."""
    return len(tokenize(row["premise"])) + len(tokenize(row["hypothesis"]))


def overlap(row):
    premise = {t.lower() for t in tokenize(row["premise"])}
    hypothesis = {t.lower() for t in tokenize(row["hypothesis"])}
    return len(hypothesis & premise) / len(hypothesis) if hypothesis else 0.0


def subsets(rows, share=0.1):
    n = round(len(rows) * share)
    out = {}
    for name, feature in (("length", length), ("overlap", overlap)):
        ranked = sorted(rows, key=feature)
        mid = (len(ranked) - n) // 2
        lo, hi = round(50 - 50 * share), round(50 + 50 * share)  # e.g. 45-55%
        out[f"shortest {share:.0%}" if name == "length" else f"lowest {name} {share:.0%}"] = ranked[:n]
        out[f"middle length {lo}-{hi}%" if name == "length" else f"middle {name} {lo}-{hi}%"] = ranked[mid:mid + n]
        out[f"longest {share:.0%}" if name == "length" else f"highest {name} {share:.0%}"] = ranked[-n:]
    return out


def accuracy(net, index, rows):
    labels = torch.tensor([r["label"] for r in rows])
    return float((predict(net, index, rows) == labels).float().mean())


def table(models, rows):
    groups = [("Validation + test (all)", "full dataset", rows)]
    for name, subset in subsets(rows).items():
        if "overlap" in name:
            values = [overlap(r) for r in subset]
            span = f"overlap {min(values):.2f}-{max(values):.2f}"
        else:
            values = [length(r) for r in subset]
            span = f"{min(values)}-{max(values)} tokens"
        groups.append((name[0].upper() + name[1:], span, subset))

    accs = {name: [] for name, _, _ in groups}
    for path in models:
        net, index = load_fold(path)
        for name, _, subset in groups:
            accs[name].append(100 * accuracy(net, index, subset))
    return [(name, span, len(subset), statistics.mean(accs[name]), statistics.stdev(accs[name]))
            for name, span, subset in groups]


def save_table(rows, path=RESULTS / "error_analysis.png"):
    columns = ["", "Pairs", "Accuracy (%)", "Range"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.axis("off")
    table = ax.table(cellText=[[name, n, f"{acc:.2f} ± {std:.2f}", span] for name, span, n, acc, std in rows],
                     colLabels=columns, colWidths=[0.34, 0.16, 0.22, 0.26], cellLoc="center", loc="center")
    table.scale(1, 2)
    table.set_fontsize(11)
    for col in range(len(columns)):
        table[1, col].set_facecolor("#dbe8fb")
    ax.set_title(f"Embed and pool, mean ± std over {len(MODELS)} folds", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"\nsaved {path}")


def main(cache_dir=DEFAULT_CACHE_DIR):
    rows = [r for split in ("validation", "test") for _, r in load_snli(split, cache_dir=cache_dir)]
    results = table(MODELS, rows)

    print(f"{'':<26}{'Pairs':>8}{'Accuracy (%)':>17}   Range")
    for name, span, n, acc, std in results:
        print(f"{name:<26}{n:>8}{f'{acc:.2f} ± {std:.2f}':>17}   {span}")
    save_table(results)


if __name__ == "__main__":
    main()
