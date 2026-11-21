"""Hyperparameter search for the ConceptNet LSTM on SNLI."""

# Cross-validation settings from Project 1
SEEDS = [42]
NUM_FOLDS = 5

# LSTM encoder with fixed ConceptNet embeddings
LSTM_HIDDEN_SIZES = [30]
LSTM_NUM_LAYERS = [3]
LSTM_BIDIRECTIONAL = [True]

# Each tuple gives the classifier's hidden layer widths in order.
CLASSIFIER_HIDDEN_SIZES = [(50, 50)]
CLASSIFIER_DROPOUT_RATES = [0.1]

# AdamW and training schedule
LEARNING_RATES = [3e-3, 1e-3, 5e-4]
WEIGHT_DECAYS = [1e-4]
BATCH_SIZES = [256, 512, 1024]
NUM_EPOCHS = [15]


if __name__ == "__main__":
    import sys
    from itertools import product
    from pathlib import Path

    import torch

    rnn_dir = Path(__file__).resolve().parent
    project_dir = rnn_dir.parents[2]
    sys.path[:0] = [str(rnn_dir.parent), str(project_dir)]
    from rnn import data, experiment

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    output_dir = rnn_dir / "output" / "conceptnet"
    corpus = None

    for seed in SEEDS:
        folds = None
        grid = product(
            LSTM_HIDDEN_SIZES, LSTM_NUM_LAYERS, LSTM_BIDIRECTIONAL,
            CLASSIFIER_HIDDEN_SIZES, CLASSIFIER_DROPOUT_RATES,
            LEARNING_RATES, WEIGHT_DECAYS, BATCH_SIZES, NUM_EPOCHS,
        )
        for hidden_size, depth, bidirectional, widths, dropout, lr, weight_decay, batch_size, epochs in grid:
            head = "-".join(map(str, widths)) or "linear"
            direction_suffix = "-bidir" if bidirectional else ""
            name = (
                f"frozen-lstm{depth}x{hidden_size}{direction_suffix}-mlp{head}-drop{dropout}"
                f"-lr{lr}-wd{weight_decay}-batch{batch_size}-epochs{epochs}-cv{NUM_FOLDS}"
            )
            run_dir = output_dir / name / f"seed-{seed}"
            try:
                run_dir.mkdir(parents=True)
            except FileExistsError:
                print(f"Skipping existing run: {run_dir}", flush=True)
                continue

            if corpus is None:
                corpus, embeddings, row_ids = data.load_snli_training(project_dir / "cache")
                table_path = data.save_numberbatch(embeddings, output_dir)
                print(f"Data ready | {len(corpus):,} SNLI pairs | device={device}", flush=True)
            if folds is None:
                folds = data.cross_validation_split(corpus, row_ids, NUM_FOLDS, seed)

            print(f"Running {name}, seed={seed}", flush=True)
            run = experiment.cross_validation(
                data_splits=folds,
                outcome_space=data.SNLI_LABELS,
                tokenizer=None,  # The corpus already contains token IDs.
                emb_model=embeddings,
                enc_type="lstm",
                enc_kwargs={
                    "hidden_size": hidden_size,
                    "num_layers": depth,
                    "bidirectional": bidirectional,
                    "freeze_embeddings": True,
                },
                regressor_kwargs={"hidden_sizes": widths, "dropout": dropout},
                opt_kwargs={
                    "learning_rate": lr,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "num_epochs": epochs,
                },
                device=device,
                seed=seed,
                shared_table=table_path,
                save_to=run_dir,
            )
            print(f"Validation accuracies: {run['accuracies']}", flush=True)
