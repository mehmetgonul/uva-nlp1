import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb

RESULTS = Path(__file__).resolve().parent / "results"

SWEEPS = [  # (CSV, title, rows, columns)
    ("width_depth", "Width x depth", "hidden_size", "num_layers"),
    ("activation_dropout", "Activation x dropout", "activation", "dropout"),
    ("lr_batch_size", "Learning rate x batch size", "lr", "batch_size"),
]
LABELS = {"hidden_size": "hidden width", "num_layers": "hidden layers", "lr": "learning rate",
          "batch_size": "batch size", "activation": "activation", "dropout": "dropout"}
ACTIVATION_ORDER = ["tanh", "relu", "gelu"]

BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#104281"])
INK, MUTED, SURFACE = "#1f1f1e", "#6b6a66", "#ffffff"


def value(text):
    try:
        return float(text)
    except ValueError:
        return text


def frmt(v):
    return v if isinstance(v, str) else f"{v:g}"


def order(values):
    return sorted(values, key=lambda v: (ACTIVATION_ORDER.index(v), 0) if v in ACTIVATION_ORDER else (-1, v))


def subtitle(rows, row_key, col_key):
    """The settings shared by the whole sweep, e.g. 'head 60 wide x 3 layers, lr 0.001, batch 256'."""
    fixed = {k: rows[0][k] for k in LABELS if k not in (row_key, col_key) and len({r[k] for r in rows}) == 1}
    parts = []
    if "hidden_size" in fixed and "num_layers" in fixed:
        parts.append(f"depth: {fixed['num_layers']}, width: {fixed['hidden_size']}")
    for k, name in (("activation", "act"), ("lr", "lr"), ("batch_size", "batch"), ("dropout", "dropout")):
        if k in fixed:
            parts.append(f"{name}: {frmt(value(fixed[k]))}")
    return ", ".join(parts)


def collect():
    tables = []
    for name, title, row_key, col_key in SWEEPS:
        path = RESULTS / f"{name}.csv"
        if path.exists():
            rows = list(csv.DictReader(path.open()))
            cells = {}
            for r in rows:
                cells.setdefault((value(r[row_key]), value(r[col_key])), []).append(float(r["val_acc"]))
            tables.append((f"{title}\n{subtitle(rows, row_key, col_key)}", LABELS[row_key], LABELS[col_key],
                           order({r for r, _ in cells}), order({c for _, c in cells}), cells))
    return tables


def draw(ax, title, row_label, col_label, rows, cols, cells, norm):
    grid = np.array([[100 * statistics.mean(cells[r, c]) for c in cols] for r in rows])
    ax.pcolormesh(grid, cmap=BLUES, norm=norm, edgecolors=SURFACE, linewidth=2)
    best = np.unravel_index(grid.argmax(), grid.shape)

    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            luminance = np.dot(to_rgb(BLUES(norm(grid[i, j]))), [0.2126, 0.7152, 0.0722])
            color = SURFACE if luminance < 0.45 else INK
            ax.text(j + 0.5, i + 0.47, f"{grid[i, j]:.2f}%", ha="center", va="bottom",
                    fontsize=12, fontweight="bold" if (i, j) == best else "normal", color=color)
            ax.text(j + 0.5, i + 0.56, f"± {100 * statistics.stdev(cells[r, c]):.2f}", ha="center", va="top",
                    fontsize=9, color=color)
    ax.add_patch(plt.Rectangle((best[1] + 0.04, best[0] + 0.04), 0.92, 0.92, fill=False, edgecolor=INK, linewidth=1.5))

    ax.set_xlim(0, len(cols))
    ax.set_ylim(0, len(rows))
    ax.set_xticks(np.arange(len(cols)) + 0.5, [frmt(c) for c in cols], color=MUTED)
    ax.set_yticks(np.arange(len(rows)) + 0.5, [frmt(r) for r in rows], color=MUTED)
    ax.set_xlabel(col_label, color=MUTED)
    ax.set_ylabel(row_label, color=MUTED)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.tick_params(length=0)
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    tables = collect()
    means = [100 * statistics.mean(a) for *_, cells in tables for a in cells.values()]
    norm = Normalize(min(means), max(means))  # one colour scale for all tables

    # sizes in inches: up to 3 panels in a row, otherwise 2 per row
    n = len(tables)
    ncols = n if n <= 3 else 2
    nrows = -(-n // ncols)
    panel_w, panel_h, left, hgap = 3.9, 2.9, 1.0, 1.5
    header, title, below = 1.0, 0.65, 0.7
    bar_gap, bar_w, right = 0.5, 0.15, 1.0
    W = left + ncols * panel_w + (ncols - 1) * hgap + bar_gap + bar_w + right
    H = header + nrows * (title + panel_h + below)
    fig = plt.figure(figsize=(W, H), facecolor=SURFACE)
    for k, table in enumerate(tables):
        row, col = divmod(k, ncols)
        x = left + col * (panel_w + hgap)
        y = H - header - row * (title + panel_h + below) - title - panel_h
        draw(fig.add_axes([x / W, y / H, panel_w / W, panel_h / H]), *table, norm)

    bar_x = left + ncols * panel_w + (ncols - 1) * hgap + bar_gap
    cax = fig.add_axes([bar_x / W, below / H, bar_w / W, (H - header - title - below) / H])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=BLUES), cax=cax)
    bar.set_label("mean val accuracy (%)", color=MUTED)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=MUTED, length=0)

    fig.text(0.2 / W, 1 - 0.35 / H, "Embed-and-pool (ConceptNet), val accuracy after 10 epochs, "
             "5-fold cross-validation (mean ± std over folds)", color=INK, fontsize=12.5, va="center")

    fig.savefig(RESULTS / "sweep_tables.png", dpi=200, facecolor=SURFACE)
    plt.show()


if __name__ == "__main__":
    main()
