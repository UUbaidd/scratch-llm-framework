# custom_dl/layers/embeddings.py
import numpy as np
from custom_dl.base import Module

class Embedding(Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.register_parameter("weight", np.random.randn(vocab_size, d_model) * 0.1)

    def forward(self, token_ids):
        return self.parameters()["weight"][token_ids]

def get_positional_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            pe[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                pe[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
    return pe