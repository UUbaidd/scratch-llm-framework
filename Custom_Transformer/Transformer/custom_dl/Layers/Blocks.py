# custom_dl/layers/blocks.py
import numpy as np
from custom_dl.base import Module
from custom_dl.Layers.linear import Linear
from custom_dl.Layers.Attention import MultiHeadAttention

def layer_norm(x, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)

class FeedForward(Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)

    def forward(self, x):
        hidden = np.maximum(0, self.linear1(x))  # ReLU
        return self.linear2(hidden)

class TransformerBlock(Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff=d_model * 4)

    def forward(self, x, mask=None):
        attn_out = self.mha(x, mask=mask)
        x = layer_norm(x + attn_out)
        
        ffn_out = self.ffn(x)
        x = layer_norm(x + ffn_out)
        return x