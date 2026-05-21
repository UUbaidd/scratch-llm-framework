# custom_dl/models/language_model.py
import numpy as np
from custom_dl.base import Module
from custom_dl.Layers.Embeddings import Embedding, get_positional_encoding
from custom_dl.Layers.Blocks import TransformerBlock
from custom_dl.Layers.linear import Linear

class LanguageModel(Module):
    def __init__(self, vocab_size, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.embeddings = Embedding(vocab_size, d_model)
        self.block = TransformerBlock(d_model, num_heads)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, token_ids):
        seq_len = len(token_ids)
        
        x = self.embeddings(token_ids)
        pe = get_positional_encoding(seq_len, self.d_model)
        x = (x + pe)[np.newaxis, :, :]  # Add batch dimension
        
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        hidden_states = self.block(x, mask=mask)
        logits = self.lm_head(hidden_states)
        
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return probs