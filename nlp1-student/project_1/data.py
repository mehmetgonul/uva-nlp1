import pandas as pd
import torch
from torch.utils.data import Dataset

splits = {'test': 'plain_text/test-00000-of-00001.parquet', 'validation': 'plain_text/validation-00000-of-00001.parquet', 'train': 'plain_text/train-00000-of-00001.parquet'}

TRAIN_SET = pd.read_parquet("hf://datasets/stanfordnlp/snli/" + splits["train"])
VALIDATION_SET = pd.read_parquet("hf://datasets/stanfordnlp/snli/" + splits["validation"])
TEST_SET = pd.read_parquet("hf://datasets/stanfordnlp/snli/" + splits["test"])

# combining train and val to do k-fold splits
TRAIN_VAL_SET = pd.concat([TRAIN_SET, VALIDATION_SET], ignore_index=True)
