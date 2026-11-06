import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from part1.experiment import download_ag_news_texts, train_tokenizer

from tokenizers import Tokenizer

path_of_tokenizer_snli = Path(__file__).resolve().parent / "snli_bpe_5k.json"

if path_of_tokenizer_snli.is_file():
    tokenizer = Tokenizer.from_file(str(path_of_tokenizer_snli))
else:
    from project_1.data import TRAIN_SET
    corpus = (TRAIN_SET["premise"].dropna().tolist() + TRAIN_SET["hypothesis"].dropna().tolist())
    tokenizer = train_tokenizer(corpus, vocab_size=5000, min_frequency=2)
    
    path_of_tokenizer_snli.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path_of_tokenizer_snli))


def tokenize(text):
    return tokenizer.encode(text)

def get_vocab_size():
    return tokenizer.get_vocab_size()
