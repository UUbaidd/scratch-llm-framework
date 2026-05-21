# custom_dl/base.py
import numpy as np

class Module:
    """
    Abstract Base Class for all neural network components.
    Tracks state and coordinates forward/backward passes.
    """
    def __init__(self):
        self._parameters = {}

    def register_parameter(self, name, value):
        self._parameters[name] = value

    def parameters(self):
        return self._parameters

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)