import numpy as np


class Dense:

    def __init__(self, input_shape, output_shape, activation, dropout=None, batch_norm_momentum=False, beta1=0.9, beta2=0.999):

        self.weights = np.random.normal(0, np.sqrt(1/input_shape), (input_shape, output_shape))
        self.biases = np.random.randn(1, output_shape)
        self.output_shape = output_shape
        self.output = np.zeros(output_shape)
        self.activation = activation
        self.dropout = dropout
        if dropout:
            self.dropout_mask = None
        self.input = None
        self.training = False
        self.regularize_para = 0
        self.mean = np.zeros_like(self.biases)
        self.variance = np.ones_like(self.biases)
        self.gamma = np.ones_like(self.biases)
        self.batch_normalize = batch_norm_momentum
        self.batch_norm_cache = None
        self.update_flag = True
        self.beta1 = beta1
        self.beta2 = beta2
        self.vw = 0
        self.sw = 0
        self.vb = 0
        self.sb = 0
        self.vg = 0
        self.sg = 0

    def forward(self, input_layer):
        self.input = input_layer
        output = np.dot(input_layer, self.weights)
        if self.batch_normalize:
            output = self.batch_norm(output)
        self.output = output + self.biases
        if self.dropout and self.training:
            mask = np.random.rand(1, self.output.shape[1]) > self.dropout
            self.dropout_mask = np.broadcast_to(mask, self.output.shape)
            self.output[self.dropout_mask] = 0
        self.output = self.activation.forward(self.output)
        return self.output

    def find_gradient(self, delta):
        delta_w = np.dot(self.input.T, delta)
        delta_z = np.dot(delta, self.weights.T)
        delta_b = np.sum(delta, axis=0, keepdims=True)
        return delta_w, delta_b, delta_z

    def update(self, delta, learning_rate, mini_size):
        delta = self.activation.backward(delta)
        if self.batch_normalize:
            delta, delta_g = self.batch_norm_backwards(delta)
            self.vg = self.vg = self.beta1 * self.vg + (1 - self.beta1) * delta_g
            self.sg = self.beta2 * self.sg + (1 - self.beta2) * np.square(delta_g)

            self.gamma -= learning_rate * self.vg / np.sqrt(self.sg + 1e-8)

        delta_w, delta_b, delta_z = self.find_gradient(delta)
        if self.update_flag:
            delta_w = delta_w + (self.regularize_para / mini_size) * self.weights
            delta_b = delta_b
            self.vw = self.beta1 * self.vw + (1 - self.beta1) * delta_w
            self.vb = self.beta1 * self.vb + (1 - self.beta1) * delta_b
            self.sw = self.beta2 * self.sw + (1 - self.beta2) * np.square(delta_w)
            self.sb = self.beta2 * self.sb + (1 - self.beta2) * np.square(delta_b)

            self.weights -= learning_rate * self.vw / np.sqrt(self.sw + 1e-8)
            self.biases -= learning_rate * self.vb / np.sqrt(self.sb + 1e-8)
        return delta_z

    def batch_norm(self, inputs):
        if self.training:
            mean = np.mean(inputs, axis=0, keepdims=True)
            variance = np.var(inputs, axis=0, keepdims=True)
            self.mean = self.batch_normalize * self.mean + (1 - self.batch_normalize) * mean
            self.variance = self.batch_normalize * self.variance + (1 - self.batch_normalize) * variance
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
