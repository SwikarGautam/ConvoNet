from convolution import Conv
from dense import Dense
from flatten import Flatten
from maxpooling import Maxpool
import random


class ConvoNet:

    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.sequence = []

    def predict(self, input_values):
        inp = input_values
        for i, layer in enumerate(self.sequence):
            layer.training = False
            inp = layer.out(inp)
            if i < len(self.sequence) - 1:
                inp = self.leaky_relu(inp)
        return self.softmax(inp)

    def train(self, training_data, epoch, learning_rate, mini_batch_size):
        for e in range(epoch):

            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in
                            range(0, len(training_data), mini_batch_size)]

            for mini_batch in mini_batches:

                x_train, y_train = [np.array(list(x)) for x in zip(*mini_batch)]
                inp = np.stack(x_train, axis=2)

                for i, layer in enumerate(self.sequence):
                    layer.training = True
                    inp = layer.out(inp)

                    if i < len(self.sequence) - 1:
                        inp = self.leaky_relu(inp)

                output = self.softmax(inp)
                print(self.cross_entropy_loss(output, y_train.squeeze(), mini_batch_size))

                for i, layer in enumerate(self.sequence[::-1]):
                    if i == 0:
                        delta = output - y_train.squeeze()
                    else:
                        delta = (delta * self.d_leaky_relu(layer.output))
                    delta = layer.update(delta, learning_rate, mini_batch_size)

    def add(self, layer_type, weight_shape=None, layer_shape=None, window_size=2, stride=1, padding=0, dropout=None,
            batch_normalize=False):
        if len(self.sequence) == 0:
            in_shape = self.input_shape
        else:
            in_shape = self.sequence[-1].output_shape
        if layer_type == 'convolve':
            self.sequence.append(Conv(in_shape, weight_shape, stride, padding, batch_normalize))
        elif layer_type == 'maxpool':
            self.sequence.append(Maxpool(in_shape, window_size, stride))
        elif layer_type == 'Dense':
            self.sequence.append(Dense(in_shape, layer_shape, dropout, batch_normalize))
        elif layer_type == 'flatten':
            self.sequence.append(Flatten(in_shape))
        else:
            print('layer type not understood')

    @staticmethod
    def cross_entropy_loss(x, x_lables, mini_size):
        return -(np.log(x) * x_lables).sum() / mini_size

    @staticmethod
    def leaky_relu(x):
        return np.maximum(0.001 * x, x)

    @staticmethod
    def d_leaky_relu(x):
        r = np.zeros_like(x)
        r[x > 0] = 1
        r[x < 0] = 0.001
        return r

    @staticmethod
    def softmax(x):
        e = np.exp(x - x.max(axis=1, keepdims=True)/2)
        return e / np.nansum(e, axis=1, keepdims=True)
