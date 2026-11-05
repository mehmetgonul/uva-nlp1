import torch

USE_PRETRAINED_TOKENIZER = False

if USE_PRETRAINED_TOKENIZER: 
    # Standard tokenizer (gpt2)
    from tokenize_standard import tokenize
else:
    # Custom tokenizer in case the standard tokenizer uses too big vocab, and takes too long to train on
    from tokenize_custom import tokenize

from data import TRAIN_SET, VALIDATION_SET, TEST_SET

# Setting the device 
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

# print()
# print("="* 50)
# print("Imports loaded, running...")
# print("="* 50)
# print()

# # Example of tokenize() use:
# premise = TRAIN_SET.loc[0, "premise"]
# hypothesis = TRAIN_SET.loc[0, "hypothesis"]

# print(tokenize(premise).ids)
# print(tokenize(hypothesis).ids)

from project_1.featurize import *
from project_1.models import model_factory
from part1.experiment import cross_validation_split
from data import TRAIN_VAL_SET

def get_feature_fn(feature: str):
    
    if feature == "bag_of_words":
        return bag_of_words
    
    if feature == "indicators":
        return token_indicators
    
    return overlap


SEED = 42
K = 5
MODEL_NAMES = ["linear", "mlp", "deep_mlp"]

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

data = TRAIN_VAL_SET.loc[
    TRAIN_VAL_SET["label"].isin([0, 1, 2])
    ].dropna(subset=["premise", "hypothesis"]).reset_index(drop=True)

targets = data["label"].to_numpy(dtype=np.int64)


inputs = []
for premise, hypothesis in zip(data["premise"], data["hypothesis"]):
    inputs.append((tokenize(premise).ids, tokenize(hypothesis).ids))

splits = cross_validation_split(
    pairs=list(zip(inputs, targets)),
    k=K,
    seed=SEED
)


vocab_size = 0

for split in splits.values():
    train_X, train_y = zip(*split["training"])
    val_X, val_y = zip(*split["validation"])
    
    for feature in FEATURES:
        X_train = prep_inputs(train_X, feature, vocab_size)
        X_val = prep_inputs(val_X, feature, vocab_size)
        
        for model_name in MODEL_NAMES:
            model = model_factory(model_name, 128, "relu")

            
            model.fit(X_train, train_y)
            acc = model.score(X_val, val_y)
            
            conf_name = f"{model_name}_{feature}"
            
            