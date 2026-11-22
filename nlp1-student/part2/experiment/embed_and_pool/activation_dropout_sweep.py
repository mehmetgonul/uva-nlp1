import itertools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from embed_and_pool import sweep

ACTIVATIONS = ["tanh", "relu", "gelu"]
DROPOUTS = [0.1, 0.2, 0.5]

sweep(
    "activation_dropout",
    [{"activation": a, "dropout": d} for a, d in itertools.product(ACTIVATIONS, DROPOUTS)],
    max_workers=7,
    hidden_size=60,
    num_layers=3,
    lr=1e-3,
    batch_size=256,
    weight_decay=1e-4,
    epochs=10,
)
