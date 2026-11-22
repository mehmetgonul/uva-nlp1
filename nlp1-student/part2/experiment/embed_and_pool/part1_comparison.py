import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent / "results"
HEADS = {"linear": "Linear Model", "mlp": "MLP Model", "deep_mlp": "Deep MLP Model"}

# last week's results, mean ± std val accuracy
LAST_WEEK = {
    "Token Indicators": [(63.67, 0.15), (70.44, 0.17), (69.79, 0.33)],
    "Bag of Words": [(63.66, 0.16), (70.36, 0.21), (69.70, 0.34)],
    "Overlap": [(68.31, 0.15), (71.37, 0.21), (71.31, 0.15)],
}

accs = {}
for r in csv.DictReader((RESULTS / "part1_heads.csv").open()):
    accs.setdefault(r["head"], []).append(100 * float(r["val_acc"]))
rows = {**LAST_WEEK, "Embed and pool (ConceptNet)": [(statistics.mean(accs[h]), statistics.stdev(accs[h])) for h in HEADS]}

print(f"{'':<30}" + "".join(f"{h:>18}" for h in HEADS.values()))
for name, cells in rows.items():
    print(f"{name:<30}" + "".join(f"{f'{m:.2f} ± {s:.2f}':>18}" for m, s in cells))

fig, ax = plt.subplots(figsize=(9, 2.4))
ax.axis("off")
table = ax.table(cellText=[[name, *(f"{m:.2f} ± {s:.2f}" for m, s in cells)] for name, cells in rows.items()],
                 colLabels=["", *HEADS.values()], colWidths=[0.34, 0.22, 0.22, 0.22], cellLoc="center", loc="center")
table.scale(1, 2)
table.set_fontsize(11)
for col in range(len(HEADS) + 1):  # highlight this week's row
    table[len(rows), col].set_facecolor("#dbe8fb")
ax.set_title("Val accuracy (%), mean ± std over 3 seeds x 5 folds", fontsize=10)
fig.tight_layout()
fig.savefig(RESULTS / "part1_comparison.png", dpi=200, bbox_inches="tight")
plt.show()
