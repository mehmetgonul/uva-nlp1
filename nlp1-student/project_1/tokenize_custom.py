import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from part1.experiment import download_ag_news_texts, train_tokenizer

corpus = download_ag_news_texts()
tokenizer = train_tokenizer(corpus, vocab_size=5000, min_frequency=2)

def tokenize(text):
    return tokenizer.encode(text)
