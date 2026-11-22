import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from embed_and_pool import RESULTS, SCORES, train

HEADS = {
    "linear": lambda d: nn.Sequential(nn.Linear(d, 3), nn.LogSoftmax(-1)),
    "mlp": lambda d: nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 3), nn.LogSoftmax(-1)),
    "deep_mlp": lambda d: nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, 256), nn.GELU(),
                                        nn.Dropout(0.2), nn.Linear(256, 3), nn.LogSoftmax(-1)),
}
SEEDS = [43, 44]
RESULTS_FILE = RESULTS / "part1_heads.csv"


def run(seed):  # the seed sets both the fold split and the initialisation
    settings = {"seed": seed, "lr": 5e-4, "batch_size": 256, "weight_decay": 1e-3, "epochs": 10}
    with open(os.devnull, "w") as f, redirect_stdout(f), redirect_stderr(f):
        train(HEADS, settings, ["head", *settings, *SCORES], RESULTS_FILE, progress=False, **settings)
    return seed


if __name__ == "__main__":
    todo = [s for s in SEEDS]
    with ProcessPoolExecutor(max_workers=len(SEEDS)) as pool:
        for seed in pool.map(run, todo):
            print(f"seed {seed} done", flush=True)
