from itertools import combinations

import pandas as pd

from part1 import analysis


def evaluate(results, rope=0.005):
    data = analysis.make_data_frame(results)
    summary = data.groupby("model")["accuracy"].agg(["mean", "std"])

    scores = data.pivot(index=["run", "fold"], columns="model", values="accuracy")
    if scores.isna().any().any():
        raise ValueError("Systems must have the same runs and folds")

    number_of_runs = data["run"].nunique()
    comparisons = []

    for system_a, system_b in combinations(scores.columns, 2):
        accuracy_a = scores[system_a]
        accuracy_b = scores[system_b]
        posterior = analysis.BenavoliCorrelatedT(accuracy_a, accuracy_b, runs=number_of_runs)

        a_wins, tie, b_wins = posterior.probs(rope)

        comparisons.append(
            {
                "system_a": system_a,
                "system_b": system_b,
                "mean_diff": (accuracy_b - accuracy_a).mean(),
                "p_a_wins": a_wins,
                "p_tie": tie,
                "p_b_wins": b_wins,
            }
        )

    comparisons = pd.DataFrame(comparisons)
    # analysis.plot_box_and_points(data) save this to a file
    
    return summary, comparisons
