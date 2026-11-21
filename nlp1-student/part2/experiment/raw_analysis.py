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

print(f"One-word substitutions in SNLI train: {sum(label_counts.values()):,}")
for label in LABELS:
    print(f"  {label}: {label_counts[label]:,}")

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

print(f"\nExample premise: {example_premise}")
print(f"{'Label':<15} {'Train row':<10} {'Hypothesis':<35} Word cosine (Numberbatch)")
vectors = model_vectors["Numberbatch"]
for label, row_id, hypothesis, before, after in example_rows:
    if before in vectors and after in vectors:
        score = 1 - cosine(vectors[before], vectors[after])
        print(f"{label:<15} {row_id:<10} {hypothesis:<35} {score:.3f}")

comparisons = {
    "Entailment vs. contradiction": ["entailment", "contradiction"],
    "Entailment vs. neutral": ["entailment", "neutral"],
    "Entailment vs. both": ["entailment", "neutral", "contradiction"],
}
print("\nHow often does the entailed word have the highest cosine with the original?")
print("Wins / distinct word sets covered by all three models (ties do not count as wins).")
print(f"{'Comparison':<30} {'Numberbatch':>18} {'GloVe':>18} {'fastText':>18}")
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
    print(f"{comparison:<30}", *scores)
