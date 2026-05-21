# custom_dl/layers/attention.py
import numpy as np
from custom_dl.base import Module
from custom_dl.Layers.linear import Linear

class MultiHeadAttention(Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Use our clean Linear modules instead of raw arrays
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.o_proj = Linear(d_model, d_model)

    def split_heads(self, x):
        batch_size, seq_len, d_model = x.shape
        return x.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        
        Q = self.split_heads(self.q_proj(x))
        K = self.split_heads(self.k_proj(x))
        V = self.split_heads(self.v_proj(x))

        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)

        if mask is not None:
            scores += (mask * -1e9) 

        weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True)) 
        weights /= np.sum(weights, axis=-1, keepdims=True)

        context = np.matmul(weights, V)
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        return self.o_proj(context)