from convolution import Conv
from dense import Dense
from flatten import Flatten
from maxpooling import Maxpool
from activations import *
from loss import *
import numpy as np
import random


class ConvoNet:

    def __init__(self, input_shape, loss):
        
        self.input_shape = input_shape[0], input_shape[1], 1, input_shape[2]
        self.regularize_para = 0
        self.sequence = []
        self.loss = loss
        self.frobenius_norm = 0

    def predict(self, input_values):
        
        inp = input_values
        for i, layer in enumerate(self.sequence):
            layer.training = False
            inp = layer.forward(inp)
        return inp

    def train(self, training_data, epoch, learning_rate, mini_batch_size, print_mini_batch_loss):

        for e in range(epoch):
            mini_batches = self.arrange_data(training_data, mini_batch_size)
            print("Generation:", e+1)

            for mini_batch in mini_batches:
                x_train, y_train = mini_batch
                self.frobenius_norm = 0
                inp = x_train
                for i, layer in enumerate(self.sequence):
                    layer.training = True
                    inp = layer.forward(inp)

                    if isinstance(layer, Dense):
                        layer.regularize_para = self.regularize_para
                        self.frobenius_norm += np.sum(layer.weights**2)

                output = inp
                if print_mini_batch_loss:
                    print('mini batch loss:', self.loss.calc_loss(output, y_train.squeeze(),
                                                                  mini_batch_size, self.frobenius_norm))
                delta = self.loss.backward(output, y_train.squeeze())
                for i, layer in enumerate(self.sequence[::-1]):
                    delta = layer.update(delta, learning_rate, mini_batch_size)

    def add(self, layer):
        self.sequence.append(layer)

    def set_regularization(self, regularization_parameter):
        self.loss.regularization = regularization_parameter
        self.regularize_para = regularization_parameter

    def set_batch_norm_momentum(self, momentum):
        for layer in self.sequence:
            layer.batch_normalize = momentum

    @staticmethod
    def arrange_data(data, mini_size):
        random.shuffle(data)
        x_train, y_train = [np.array(list(x)) for x in zip(*data)]
        inp = np.stack(x_train, axis=2)
        x_train = np.array_split(inp, (len(data)+1)//mini_size, axis=2)
        y_train = np.array_split(y_train, (len(data)+1)//mini_size, axis=0)
        return list(zip(x_train, y_train))

    def evaluate(self, test_set):
        
        x_test, y_test = [np.array(list(x)) for x in zip(*test_set)]
        inp = np.stack(x_test, axis=2)
        predictions = self.predict(inp)
        correct = np.count_nonzero(np.argmax(predictions, axis=1) == np.argmax(y_test.squeeze(), axis=1))
        loss = self.loss.calc_loss(predictions, y_test.squeeze(), len(test_set), self.frobenius_norm)
        accuracy = correct * 100.0 / len(test_set)
        print('loss:', loss)
        print('accuracy:', accuracy, '%')

    def get_flattened_shape(self):
        input_layer = np.zeros(self.input_shape)
        last_layer = self.predict(input_layer)
        return last_layer.flatten().size
