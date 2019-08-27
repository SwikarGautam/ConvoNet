import numpy as np


class LeakyReLu:
    
    def __init__(self, alpha=0.001):
        self.alpha = alpha
        self.input = None

    def forward(self, z):
        self.input = z
        return np.maximum(self.alpha * z, z)

    def backward(self, d_a):
        r = np.zeros_like(self.input)
        r[self.input > 0] = 1
        r[self.input < 0] = self.alpha
        return r * d_a


class Sigmoid:
    
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, z):
        self.input = z
        self.output = 1 / (1 + np.exp(-z))
        return self.output

    def backward(self, d_a):
        return (self.output * (1 - self.output)) * d_a


class Tanh:

    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, z):
        self.output = np.tanh(z)
        return self.output

    def backward(self, d_a):
        return (1 - self.output ** 2) * d_a


class Softmax:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, x):
        self.input = x
        e = np.exp(x - x.max(axis=1, keepdims=True))
        self.output = e / np.nansum(e, axis=1, keepdims=True)
        return self.output

    def backward(self, d_a):
        return self.output*(d_a-np.sum(self.output*d_a, axis=1, keepdims=True))

