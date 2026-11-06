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


if __name__ == "__main__":
    import json
    from pathlib import Path
    from statistics import mean, stdev
    import numpy as np
    
    results_path = Path(__file__).resolve().parents[1] / "results"
    SEEDS = [42]
    K = 5
    
    res = {}
    
    for model_confg_dir in (results_path.iterdir()):
        runs = []
        
        # order follos the folder path construction
        for seed in SEEDS:
            accs = {}
            
            for fold_num in range(K):
                json_path = (model_confg_dir / f"seed-{seed}" / f"fold-{fold_num}" / "metric.json")
                
                with json_path.open(encoding="utf-8") as f:
                    metric = json.load(f)
                    
                accs[fold_num] = float(metric["acc"])
                
            runs.append({"seed": seed, "accuracies": accs})
            
            
        
        res[model_confg_dir.name] = runs
        
    for model_name, runs in res.items():
        global_avg = 0.
        all_fold = []
        
        for per_seed in runs:
            fold_score = list(per_seed["accuracies"].values())
            global_avg += (sum(fold_score) / len(fold_score))
            all_fold.extend(fold_score)
            
        global_avg /= len(runs)
        fold_std = np.std(all_fold, ddof=1)
        
        print(f"{model_name}: avg {global_avg * 100:.2f}% std {fold_std * 100:.2}")


    print("rope 0.005")
    summary, comparisons = evaluate(res, rope=0.005)
    
    print(comparisons.to_string(index=False, float_format="{:.6f}".format))
    
    
    