import itertools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from embed_and_pool import sweep 

LRS = [5e-4, 1e-3, 3e-3]
BATCH_SIZES = [128, 256, 512]

sweep(
    "lr_batch_size",
    [{"lr": lr, "batch_size": bs} for lr, bs in itertools.product(LRS, BATCH_SIZES)],
    max_workers=7,
    hidden_size=60,
    num_layers=3,
    activation="relu",
    dropout=0.1,
    weight_decay=1e-4,
    epochs=10,
)
