import argparse
import re
import sys
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cosine

WORD_PATTERN = re.compile(r"[^\W_]+(?:'[^\W_]+)*")
APOSTROPHES = str.maketrans("’‘ʼ", "'''")

project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_dir))
from part2.lib.dataset import LABELS, load_snli

parser = argparse.ArgumentParser()
parser.add_argument("--loss-examples", action="store_true", help="Save the Numberbatch loss tables as plain text")
args = parser.parse_args()

# Pinned SNLI train row IDs for the examples selected for the report.
# Each tuple contains the entailment row and its competing hypothesis row.
LOSS_EXAMPLES = {
    "Contradiction": [(245216, 245214), (387065, 387063), (174880, 174879), (3995, 3994), (29813, 29811)],
    "Neutral": [(267337, 267339), (52990, 52991), (189012, 189013), (72866, 72865), (226192, 226193)],
}
loss_row_ids = {row_id for cases in LOSS_EXAMPLES.values() for case in cases for row_id in case}
loss_rows = {}


def highlight(text, position):
    match = list(WORD_PATTERN.finditer(text.translate(APOSTROPHES)))[position]
    start, end = match.span()
    return f"{text[:start]}**{text[start:end]}**{text[end:]}"


label_counts = Counter()
replacements = {}
words = set()
example_premise = "A woman is handing out fruit."
example_rows = []

for row_id, pair in load_snli(split="train"):
    premise = WORD_PATTERN.findall(pair["premise"].lower().translate(APOSTROPHES))
    hypothesis = WORD_PATTERN.findall(pair["hypothesis"].lower().translate(APOSTROPHES))

    # Keep every pair where exactly one word was replaced.
    if len(premise) != len(hypothesis):
        continue
    changed_positions = [i for i in range(len(premise)) if premise[i] != hypothesis[i]]
    if len(changed_positions) != 1:
        continue
    position = changed_positions[0]
    before = premise[position]
    after = hypothesis[position]
    label = LABELS[pair["label"]]
    label_counts[label] += 1

    # Group hypotheses that change the same word in the same premise.
    key = (pair["premise"], position, before)
    if key not in replacements:
        replacements[key] = {name: set() for name in LABELS}
    replacements[key][label].add(after)
    words.update([before, after])
    if pair["premise"] == example_premise:
        example_rows.append((label, row_id, pair["hypothesis"], before, after))
    if args.loss_examples and row_id in loss_row_ids:
        loss_rows[row_id] = (pair["premise"], pair["hypothesis"], position, before, after)

lines = [f"One-word substitutions in SNLI train: {sum(label_counts.values()):,}"]
for label in LABELS:
    lines.append(f"  {label}: {label_counts[label]:,}")

cache_dir = project_dir / "cache" / "snli_embedding_similarity"
models = ["Numberbatch", "GloVe", "fastText"]
common_words = {word for word in words if word.isalpha()}
model_vectors = {}
for model in models:
    with np.load(cache_dir / f"{model.lower()}.npz") as cache:
        vectors = {
            word: vector
            for word, vector, direct in zip(cache["words"], cache["vectors"], cache["direct"])
            if direct and word in common_words
        }
    model_vectors[model] = vectors
    common_words.intersection_update(vectors)

lines.append(f"\nExample premise: {example_premise}")
lines.append(f"{'Label':<15} {'Train row':<10} {'Hypothesis':<35} Word cosine (Numberbatch)")
vectors = model_vectors["Numberbatch"]
for label, row_id, hypothesis, before, after in example_rows:
    if before in vectors and after in vectors:
        score = 1 - cosine(vectors[before], vectors[after])
        lines.append(f"{label:<15} {row_id:<10} {hypothesis:<35} {score:.3f}")

comparisons = {
    "Entailment vs. contradiction": ["entailment", "contradiction"],
    "Entailment vs. neutral": ["entailment", "neutral"],
    "Entailment vs. both": ["entailment", "neutral", "contradiction"],
}
lines.extend([
    "",
    "How often does the entailed word have the highest cosine with the original?",
    "Wins / distinct word sets covered by all three models (ties do not count as wins).",
    f"{'Comparison':<30} {'Numberbatch':>18} {'GloVe':>18} {'fastText':>18}",
])
for comparison, labels in comparisons.items():
    cases = set()
    for (_, _, original), by_label in replacements.items():
        if original not in common_words:
            continue
        choices = []
        for label in labels:
            candidates = by_label[label] & common_words
            # Skip a word if it has two of the labels we are comparing here.
            for other in labels:
                if other != label:
                    candidates -= by_label[other]
            choices.append(candidates)
        for replacements_for_case in product(*choices):
            cases.add((original, *replacements_for_case))

    scores = []
    for model in models:
        vectors = model_vectors[model]
        wins = 0
        for original, entailed, *others in cases:
            entailed_score = 1 - cosine(vectors[original], vectors[entailed])
            other_scores = [1 - cosine(vectors[original], vectors[word]) for word in others]
            if entailed_score > max(other_scores):
                wins += 1
        result = f"{wins}/{len(cases)} ({wins / len(cases):.1%})" if cases else "0/0"
        scores.append(result.rjust(18))
    lines.append(" ".join([f"{comparison:<30}", *scores]))

results_dir = Path(__file__).resolve().parent / "results"
results_dir.mkdir(exist_ok=True)
(results_dir / "raw_analysis.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

if args.loss_examples:
    vectors = model_vectors["Numberbatch"]
    lines = ["Numberbatch loss examples", "", "Scores are cosine similarities between the highlighted words."]
    for label, sample in LOSS_EXAMPLES.items():
        lines.extend(["", f"Entailment vs. {label.lower()}", ""])
        rows = [("Premise", "Hypothesis — entailment", f"Hypothesis — {label.lower()}")]
        for entailment_id, competing_id in sample:
            premise, entailment, position, original, entailed = loss_rows[entailment_id]
            _, competing, _, _, replacement = loss_rows[competing_id]
            entailment_score = 1 - cosine(vectors[original], vectors[entailed])
            competing_score = 1 - cosine(vectors[original], vectors[replacement])
            rows.append((
                highlight(premise, position),
                f"{highlight(entailment, position)} ({entailment_score:.3f})",
                f"{highlight(competing, position)} (**{competing_score:.3f}**)",
            ))
        premise_width = max(len(row[0]) for row in rows)
        entailment_width = max(len(row[1]) for row in rows)
        for premise, entailment, competing in rows:
            lines.append(f"{premise:<{premise_width}}  {entailment:<{entailment_width}}  {competing}")
    output = results_dir / "numberbatch-loss-examples.txt"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
