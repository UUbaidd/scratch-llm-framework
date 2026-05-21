# custom_dl/layers/linear.py
import numpy as np
from custom_dl.base import Module

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Industry standard initialization (Xavier/Glorot)
        bound = 1.0 / np.sqrt(in_features)
        self.register_parameter("W", np.random.uniform(-bound, bound, (in_features, out_features)))
        self.register_parameter("b", np.zeros(out_features))

    def forward(self, x):
        params = self.parameters()
        return np.dot(x, params["W"]) + params["b"]