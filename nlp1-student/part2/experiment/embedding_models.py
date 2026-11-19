"""Compare sentence embedding models on SNLI pairs."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy.spatial.distance import cosine
from sklearn.neighbors import KernelDensity

from part2.lib.dataset import DEFAULT_CACHE_DIR, FIELDS, LABELS, load_snli
from part2.lib.embeddings import CONCEPTNET, MODELS, embed_snli_pairs

COLORS = ("#0072B2", "#007A58", "#D55E00")


def mean_pool(word_vectors):
    if len(word_vectors) == 0:
        return None
    return word_vectors.mean(axis=0)


def cosine_similarity(premise, hypothesis):
    if premise is None or hypothesis is None:
        return None
    if not np.any(premise) or not np.any(hypothesis):
        return None
    return float(1 - cosine(premise, hypothesis))


def score_pair(model, row_id, label, premise, hypothesis, *, word_counts):
    premise_tokens, premise_hits, premise_prefix, hypothesis_tokens, hypothesis_hits, hypothesis_prefix = word_counts
    return {
        "model": model,
        "snli_row": row_id,
        "label": LABELS[label],
        "cosine": cosine_similarity(premise, hypothesis),
        "premise_tokens": premise_tokens,
        "premise_hits": premise_hits,
        "hypothesis_tokens": hypothesis_tokens,
        "hypothesis_hits": hypothesis_hits,
        "premise_prefix_matches": premise_prefix,
        "hypothesis_prefix_matches": hypothesis_prefix,
    }


def pooled_word_pairs(model, cache, pairs):
    embeddings = embed_snli_pairs(pairs, model, cache_dir=cache)
    for row_id, label, premise, hypothesis in embeddings:
        premise_vector = mean_pool(premise.values)
        hypothesis_vector = mean_pool(hypothesis.values)
        counts = (
            premise.token_count, premise.matched_tokens, premise.prefix_matches,
            hypothesis.token_count, hypothesis.matched_tokens, hypothesis.prefix_matches,
        )
        yield row_id, label, premise_vector, hypothesis_vector, counts


def score_model(name, pairs):
    for row_id, label, premise, hypothesis, counts in pairs:
        yield score_pair(name, row_id, label, premise, hypothesis, word_counts=counts)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_oov_statistics(path, oov_counts):
    with path.open("w", encoding="utf-8") as file:
        file.write("Out-of-vocabulary statistics — SNLI test set\n\n")
        file.write(
            f"{'Model':<25} {'Tokens':>10} {'Missing from vocab':>19}"
            f" {'Matched by prefix':>18} {'Still unmatched':>16}\n"
        )
        for model, (tokens, unmatched, prefixes) in oov_counts.items():
            missing = unmatched + prefixes
            rate = missing / tokens if tokens else 0
            missing_text = f"{missing:,} ({rate:.2%})"
            prefix_text = f"{prefixes:,}" if model == CONCEPTNET.name else "—"
            file.write(
                f"{model:<25} {tokens:>10,} {missing_text:>19}"
                f" {prefix_text:>18} {unmatched:>16,}\n"
            )
        file.write(
            "\nNumberbatch omits tokens with no prefix match from the sentence mean.\n"
            "The other models use their global <unk> vector for tokens missing from the vocabulary.\n"
        )


def report_cosines(output, scores):
    output.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    scores_by_model = defaultdict(list)
    for row in scores:
        groups[row["model"], row["label"]].append(row)
        scores_by_model[row["model"]].append(row)
    model_names = scores_by_model.keys()
    oov_counts = {}
    for model, rows in scores_by_model.items():
        tokens = sum(row[f"{field}_tokens"] for row in rows for field in FIELDS)
        hits = sum(row[f"{field}_hits"] for row in rows for field in FIELDS)
        prefixes = sum(row[f"{field}_prefix_matches"] for row in rows for field in FIELDS)
        oov_counts[model] = (tokens, tokens - hits, prefixes)
    row_count = (len(model_names) + 1) // 2
    figure = Figure(figsize=(11, 3.5 * row_count), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(row_count, 2, sharex=True, sharey=True, squeeze=False)
    if len(model_names) % 2:
        axes[-1, 1].set_visible(False)
        if row_count > 1:
            axes[-2, 1].tick_params(labelbottom=True)
    cosine_grid = np.linspace(0, 1, 400)
    summary = []
    for model, ax in zip(model_names, axes.flat):
        for label, color in zip(LABELS, COLORS):
            group = groups[model, label]
            cosines = [row["cosine"] for row in group if row["cosine"] is not None]
            scored = len(cosines)
            if scored:
                kde = KernelDensity(bandwidth=0.02).fit(np.asarray(cosines)[:, None])
                density = np.exp(kde.score_samples(cosine_grid[:, None]))
                ax.plot(cosine_grid, density, color=color, lw=1.7, label=label)
            statistics = {
                "model": model,
                "label": label,
                "pairs": len(group),
                "scored": scored,
                "mean_cosine": f"{np.mean(cosines):.4f}" if scored else "",
                "median_cosine": f"{np.median(cosines):.4f}" if scored else "",
            }
            for field in FIELDS:
                hits = sum(row[f"{field}_hits"] or 0 for row in group)
                tokens = sum(row[f"{field}_tokens"] or 0 for row in group)
                statistics[f"{field}_coverage"] = f"{hits / tokens:.4f}" if tokens else ""
            summary.append(statistics)
        tokens, unmatched, prefixes = oov_counts[model]
        rate = unmatched / tokens if tokens else 0
        subtitle = f"Unmatched: {unmatched:,} ({rate:.2%})"
        if model == CONCEPTNET.name:
            subtitle += f" · {prefixes:,} prefix matches"
        ax.set(title=f"{model}\n{subtitle}", xlim=(0, 1))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False)

    axes[0, 0].set_ylim(bottom=0)
    figure.suptitle("Premise–hypothesis similarity on SNLI test")
    figure.supxlabel("Cosine similarity between sentence vectors")
    figure.supylabel("Probability density")
    figure.savefig(output / "cosine_by_label.png", dpi=180)

    for filename, rows in (("cosine_scores.csv", scores), ("summary.csv", summary)):
        write_csv(output / filename, rows)
    write_oov_statistics(output / "oov_statistics.txt", oov_counts)


def score_static_models(cache, pairs):
    scores = []
    for model in MODELS:
        embeddings = pooled_word_pairs(model, cache, pairs)
        scores.extend(score_model(model.name, embeddings))
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()

    pairs = load_snli(cache_dir=args.cache_dir)
    scores = score_static_models(args.cache_dir, pairs)
    report_cosines(args.output_dir, scores)


if __name__ == "__main__":
    main()
