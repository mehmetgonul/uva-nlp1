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

print()
print("="* 50)
print("Imports loaded, running...")
print("="* 50)
print()

# Example of tokenize() use:
premise = TRAIN_SET.loc[0, "premise"]
hypothesis = TRAIN_SET.loc[0, "hypothesis"]

print(tokenize(premise).ids)
print(tokenize(hypothesis).ids)