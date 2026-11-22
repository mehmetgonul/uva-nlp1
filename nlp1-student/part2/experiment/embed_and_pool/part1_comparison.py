import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent / "results"


def cells(name, keys):
    out = {}
    for r in csv.DictReader((RESULTS / f"{name}.csv").open()):
        out.setdefault(tuple(r[k] for k in keys), []).append(100 * float(r["val_acc"]))
    return out


part1 = cells("part1_heads", ["head"])
best = max(cells("lr_batch_size", ["lr", "batch_size"]).values(), key=statistics.mean)
bars = {
    "part 1\nlinear": part1["linear",],
    "part 1\nmlp": part1["mlp",],
    "part 1\ndeep mlp": part1["deep_mlp",],
    "tutorial\ntanh, dropout 0.5": cells("width_depth", ["hidden_size", "num_layers"])["60", "3"],
    "tutorial\nrelu, dropout 0.1": cells("activation_dropout", ["activation", "dropout"])["relu", "0.1"],
    "tutorial\nrelu, best lr x bs": best,
}

means = [statistics.mean(a) for a in bars.values()]
stds = [statistics.stdev(a) for a in bars.values()]
colors = ["tab:gray"] * 3 + ["tab:blue"] * 3

fig, ax = plt.subplots(figsize=(9, 5))
rects = ax.bar(bars.keys(), means, yerr=stds, capsize=4, color=colors)
ax.bar_label(rects, [f"{m:.2f}%" for m in means], padding=6)
ax.set_ylim(60, 73)
ax.set_ylabel("val accuracy (%)")
ax.set_title("Embed-and-pool (ConceptNet), 10 epochs, 5-fold CV (mean ± std)")
fig.tight_layout()
fig.savefig(RESULTS / "part1_comparison.png", dpi=200)
plt.show()
