import torch
from datetime import datetime

def get_timestamp():
    return datetime.now().strftime("%H:%M:%S")


print()
print("="* 50)
print(get_timestamp(), "Starting")
print("="* 50)
print()

USE_PRETRAINED_TOKENIZER = False


def main():
    if USE_PRETRAINED_TOKENIZER: 
        # Standard tokenizer (gpt2)
        from project_1.tokenize_standard import tokenize, get_vocab_size
    else:
        # Custom tokenizer in case the standard tokenizer uses too big vocab, and takes too long to train on
        from project_1.tokenize_custom import tokenize, get_vocab_size

    from project_1.data import TRAIN_SET
    import numpy as np

    # Setting the device 
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    # # Example of tokenize() use:
    # premise = TRAIN_SET.loc[0, "premise"]
    # hypothesis = TRAIN_SET.loc[0, "hypothesis"]

    # print(tokenize(premise).ids)
    # print(tokenize(hypothesis).ids)

    from project_1.featurize import token_indicators, bag_of_words, overlap
    from project_1.model_factory import model_factory
    from part1.experiment import cross_validation_split
    from project_1.experiment_fork import cross_validation_split, cross_validation_experiment


    print()
    print("-"* 25)
    print(get_timestamp(), "Imports loaded, running...")
    print("-"* 25)
    print()

    def get_feature_fn(feature: str):
        
        if feature == "bag_of_words":
            return bag_of_words
        
        if feature == "indicators":
            return token_indicators
        
        return overlap


    SEEDS = [42]
    K = 5

    MODEL_NAMES = ["linear"]

    FEATURES = ["bag_of_words", "indicators", "overlap"]


    def prep_inputs(tokens, feature, vocab_size):
        feature_fn = get_feature_fn(feature)
        
        X = []
        
        for premise, hypothesis in tokens:
            if feature == "overlap":
                sample = feature_fn((premise, hypothesis), vocab_size)
                
            else:
                sample = np.concatenate([
                    feature_fn(premise, vocab_size), # featurized seperately
                    feature_fn(hypothesis, vocab_size), # featurized seperately
                ]) # two seperate features are concated
                
                
            X.append(sample)
            
        return X

    class FeatureEncoder:
        def __init__(self, feature, vocab_size):
            self.feature = feature
            self.vocab_size = vocab_size
            
        def __len__(self):
            enc_dim = (3 if self.feature == "overlap" else 2) * self.vocab_size
            return enc_dim
        
        def __call__(self, input):
            return prep_inputs([input], self.feature, self.vocab_size)[0]

    data = TRAIN_SET.loc[
        TRAIN_SET["label"].isin([0, 1, 2])
        ].dropna(subset=["premise", "hypothesis"]).reset_index(drop=True)

    targets = data["label"].to_numpy(dtype=np.int64)

    # test = TEST_SET.loc[
    #     TEST_SET["label"].isin([0, 1, 2])
    #     ].dropna(subset=["premise", "hypothesis"]).reset_index(drop=True)

    # test_targets = test["label"].to_numpy(dtype=np.int64)

    # test_X = []
    # for premise, hypothesis in zip(test["premise"], test["hypothesis"]):
    #     test_X.append((tokenize(premise).ids, tokenize(hypothesis).ids))

    inputs = []
    for premise, hypothesis in zip(data["premise"], data["hypothesis"]):
        inputs.append((tokenize(premise).ids, tokenize(hypothesis).ids))

    splits = {seed: cross_validation_split(
        pairs=list(zip(inputs, targets)),
        k=K,
        seed=seed
    ) for seed in SEEDS}

    print("-"* 25)
    print(get_timestamp(), "Data ready" )
    print("-"* 25)
    print()


    vocab_size = get_vocab_size()

    results = {}

    for seed in SEEDS:
        for feature in FEATURES:
            encoder = FeatureEncoder(feature, vocab_size)

            for model_name in MODEL_NAMES:
                conf_name = f"{model_name}_{feature}"
                print(get_timestamp(), f"running {conf_name}, seed={seed}")

                run = cross_validation_experiment(
                    data_splits=splits[seed],
                    outcome_space=[0, 1, 2],
                    encoder=encoder,
                    model_option=model_name,
                    save_to=f"./results/{conf_name}/seed-{seed}",
                    seed=seed,
                    smlp_args={
                        "hidden_dim": 128,
                        "dropout": 0.1,
                        "learning_rate": 5e-4,
                        "batch_size": 256,
                        "weight_decay": 1e-3,
                        "num_epochs": 10,
                        "device": device,
                    },
                )

                results.setdefault(conf_name, []).append(run)
                print(run["accuracies"])


if __name__ == "__main__":
    main()