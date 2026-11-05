import datasets as hf
import pandas as pd

splits = {'test': 'plain_text/test-00000-of-00001.parquet', 'validation': 'plain_text/validation-00000-of-00001.parquet', 'train': 'plain_text/train-00000-of-00001.parquet'}

TRAIN_SET = hf.load_dataset("stanfordnlp/snli", split="train", cache_dir='./cache').to_pandas()
