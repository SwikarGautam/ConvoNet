import numpy as np

class Flatten:

    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape[0] * input_shape[1]*input_shape[3]
        self.output = None

    def out(self, input_layer):
        self.input_shape = input_layer.shape
        i0, i1, i2, i3 = input_layer.shape
        self.output = np.array(np.array_split(input_layer, input_layer.shape[2], axis=2)).reshape(i2, i0 * i1 * i3)
        return self.output

    def update(self, delta, learning_rate):
        d = np.array_split(delta, delta.shape[0], axis=0)
        d = np.array(d).reshape((self.input_shape[2], self.input_shape[0], self.input_shape[1], 1, self.input_shape[3]))
        delta = np.stack(d, axis=2)
        return delta.squeeze(axis=3)
