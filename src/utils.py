import torch

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def build_vocab(text):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos, len(chars)

def one_hot(indices, vocab_size):
    return torch.eye(vocab_size)[indices]
