# custom_dl/optim/sgd.py
class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self, gradients):
        """
        Expects gradients to be a dictionary mapping parameter names to raw arrays.
        """
        for name, grad in gradients.items():
            if name in self.parameters:
                self.parameters[name] -= self.lr * grad