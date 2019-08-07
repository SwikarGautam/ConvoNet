import numpy as np

class Dense:

    def __init__(self, input_shape, output_shape, dropout=None):
        self.weights = np.random.randn(input_shape, output_shape)
        self.biases = np.random.randn(1, output_shape)
        self.output_shape = output_shape
        self.output = np.zeros(output_shape)
        self.dropout = dropout
        if dropout:
            self.dropout_mask = None
        self.input = None
        self.vw = 0
        self.sw = 0
        self.vb = 0
        self.sb = 0
        
    def out(self, input_layer):
        self.input = input_layer
        self.output = np.dot(input_layer, self.weights) + self.biases
        if self.dropout:
            self.dropout_mask = np.random.rand(*self.output.shape) > self.dropout
            self.output[self.dropout_mask] = 0
        return self.output

    def find_gradient(self, delta):
        if self.dropout:
            delta[self.dropout_mask] = 0
        delta_w = np.dot(self.input.T, delta)
        delta_z = np.dot(delta, self.weights.T)
        delta_b = np.sum(delta, axis=0, keepdims=True)
        return delta_w, delta_b, delta_z

    def update(self, delta, learning_rate, beta1=0.9, beta2=0.999):
        delta_w, delta_b, delta_z = self.find_gradient(delta)
        self.vw = beta1 * self.vw + (1 - beta1) * delta_w
        self.vb = beta1 * self.vb + (1 - beta1) * delta_b
        self.sw = beta2 * self.sw + (1 - beta2) * np.square(delta_w)
        self.sb = beta2 * self.sb + (1 - beta2) * np.square(delta_b)

        self.weights -= learning_rate * self.vw / np.sqrt(self.sw + 1e-8)
        self.biases -= learning_rate * self.vb / np.sqrt(self.sb + 1e-8)
        return delta_z
