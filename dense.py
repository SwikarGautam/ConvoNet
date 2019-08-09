import numpy as np

class Dense:

    def __init__(self, input_shape, output_shape, dropout=None, batch_normalize=False):

        self.weights = np.random.randn(input_shape, output_shape)
        self.biases = np.random.randn(1, output_shape)
        self.output_shape = output_shape
        self.output = np.zeros(output_shape)

        self.dropout = dropout
        if dropout:
            self.dropout_mask = None
        self.input = None
        self.training = False

        self.mean = np.zeros_like(self.biases)
        self.variance = np.zeros_like(self.biases)
        self.gamma = np.ones_like(self.biases)
        self.batch_normalize = batch_normalize
        self.batch_norm_cache = None
        self.vw = 0
        self.sw = 0
        self.vb = 0
        self.sb = 0
        self.vg = 0
        self.sg = 0

    def out(self, input_layer):
        self.input = input_layer
        output = np.dot(input_layer, self.weights) + self.biases
        if self.batch_normalize:
            output = self.batch_norm(output)
        self.output = output + self.biases
        if self.dropout and self.training:
            mask = np.random.rand(1, self.output.shape[1]) > self.dropout
            self.dropout_mask = np.broadcast_to(mask, self.output.shape)
            self.output[self.dropout_mask] = 0
        return self.output

    def find_gradient(self, delta):
        delta_w = np.dot(self.input.T, delta)
        delta_z = np.dot(delta, self.weights.T)
        delta_b = np.sum(delta, axis=0, keepdims=True)
        return delta_w, delta_b, delta_z

    def update(self, delta, learning_rate, mini_size, beta1=0.9, beta2=0.999):
        if self.batch_normalize:
            delta, delta_g = self.batch_norm_backwards(delta)
            self.vg = self.vg = beta1 * self.vg + (1 - beta1) * delta_g
            self.sg = beta2 * self.sg + (1 - beta2) * np.square(delta_g)

            self.gamma -= learning_rate * self.vg / np.sqrt(self.sg + 1e-8)

        delta_w, delta_b, delta_z = self.find_gradient(delta)
        delta_w = delta_w / mini_size
        delta_b = delta_b / mini_size
        self.vw = beta1 * self.vw + (1 - beta1) * delta_w
        self.vb = beta1 * self.vb + (1 - beta1) * delta_b
        self.sw = beta2 * self.sw + (1 - beta2) * np.square(delta_w)
        self.sb = beta2 * self.sb + (1 - beta2) * np.square(delta_b)

        self.weights -= learning_rate * self.vw / np.sqrt(self.sw + 1e-8)
        self.biases -= learning_rate * self.vb / np.sqrt(self.sb + 1e-8)
        return delta_z

    def batch_norm(self, inputs):
        alpha = 0.99
        if self.training:
            mean = np.mean(inputs, axis=0, keepdims=True)
            variance = np.var(inputs, axis=0, keepdims=True)
            self.mean = alpha * self.mean + (1 - alpha) * mean
            self.variance = alpha * self.variance + (1 - alpha) * variance
        else:
            mean = self.mean
            variance = self.variance
        x_hat = (inputs - mean)/np.sqrt(variance + 1e-8)
        self.batch_norm_cache = x_hat, variance
        return x_hat*self.gamma

    def batch_norm_backwards(self, delta):
        x_hat, variance = self.batch_norm_cache

        d_gamma = np.sum(x_hat * delta, axis=0, keepdims=True)

        m = x_hat.shape[0]

        d_xhat = delta * self.gamma
        d_z = (m * d_xhat - np.sum(d_xhat, axis=0, keepdims=True) - x_hat
               * np.sum(d_xhat * x_hat, axis=0, keepdims=True)) / (m * np.sqrt(variance + 1e-8))

        return d_z, d_gamma

