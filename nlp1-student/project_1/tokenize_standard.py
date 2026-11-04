from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained("gpt2")

def tokenize(text):
    return tokenizer.encode(text)
