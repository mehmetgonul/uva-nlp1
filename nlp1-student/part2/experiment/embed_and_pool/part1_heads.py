import sys
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

if __name__ == "__main__":
    settings = {"lr": 5e-4, "batch_size": 256, "weight_decay": 1e-3, "epochs": 10}
    train(HEADS, settings, ["head", *settings, *SCORES], RESULTS / "part1_heads.csv", **settings)
