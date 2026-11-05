from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained("gpt2")

def tokenize(text):
    return tokenizer.encode(text)

def get_vocab_size():
    return tokenizer.get_vocab_size()
