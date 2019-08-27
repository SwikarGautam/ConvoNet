import numpy as np


class CrossEntropyLoss:
    def __init__(self, classification=True):
        self.regularization = 0
        self.classify = classification

    def calc_loss(self, x, x_labels, batch_size, norm):
        if self.classify:
            return -(np.log(x[x_labels.astype(bool)])).sum() / batch_size + (self.regularization * norm)/(2*batch_size)
        else:
            return -(x_labels*np.log(x)).sum()/batch_size + (self.regularization * norm)/(2*batch_size)

    @staticmethod
    def backward(outputs, labels):
        return - labels / outputs


class BinaryCrossEntropyLoss:
    def __init__(self, regularization):
        self.regularization = regularization

    def calc_loss(self, x, x_labels, batch_size, norm):
        return -(x*np.log(x_labels) + (1-x)*np.log(1-x_labels)).sum() / batch_size + (self.regularization * norm)/(2*batch_size)

    @staticmethod
    def backward(outputs, labels):
        return - labels / outputs + (1-labels)/(1-outputs)


class NoLoss:
    @staticmethod
    def backward(labels):
        return labels
