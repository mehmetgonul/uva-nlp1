import itertools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from embed_and_pool import sweep

WIDTHS = [40, 50, 60]
LAYERS = [1, 2, 3]

sweep(
    "width_depth",
    [{"hidden_size": w, "num_layers": n} for w, n in itertools.product(WIDTHS, LAYERS)],
    max_workers=7,
    lr=1e-3,
    batch_size=256,
    weight_decay=1e-4,
    dropout=0.5,
    activation="tanh",
    epochs=10,
)
