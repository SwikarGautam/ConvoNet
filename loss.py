import numpy as np


class CrossEntropyLoss:
    def __init__(self, classification=True):
        self.regularization = 0
        self.classify = classification

    def calc_loss(self, x, x_labels, batch_size, norm):
        if self.classify:
            return -(np.log(x[x_labels.astype(bool)])).sum() / batch_size + (self.regularization * norm)/(2*batch_size)
        else:
            x_labels = np.expand_dims(x_labels, axis=1)
            return -(x_labels*np.log(x)).sum()/batch_size + (self.regularization * norm)/(2*batch_size)

    @staticmethod
    def backward(outputs, labels):
        if labels.ndim == 1:
            labels = np.expand_dims(labels, axis=1)
        return - labels / outputs


class BinaryCrossEntropyLoss:
    def __init__(self):
        self.regularization = 0

    def calc_loss(self, x, x_labels, batch_size, norm=0):
        x_labels = np.expand_dims(x_labels, axis=1)
        return -(x_labels*np.log(x) + (1-x_labels)*np.log(1-x)).sum() / batch_size + (self.regularization * norm)/(2*batch_size)


    @staticmethod
    def backward(outputs, labels):
        if labels.ndim == 1:
            labels = np.expand_dims(labels, axis=1)
        return - labels / outputs + (1-labels)/(1-outputs)


class NoLoss:
    @staticmethod
    def backward(labels):
        return labels
