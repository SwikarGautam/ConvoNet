import tensorflow as tf  # It is only used for importing dataset
from ConvoNet import *


def vec(w):  # One hot encodes the labels
    e = np.zeros((10, 1))
    e[w] = 1.0
    return e


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

x_train = x_train.reshape((60000, 28, 28, 1)) # it should be in format (nsamples, height, width, nchannel)
x_test = x_test.reshape((10000, 28, 28, 1))
y_train.reshape(60000, 1)
y_test.reshape(10000, 1)
y_train = [vec(y) for y in y_train]
y_test = [vec(y) for y in y_test]
train = zip(x_train, y_train)  # y_train should be one hot encoded
train_set = list(train)
test = zip(x_test, y_test)
test_set = list(test) # its fromat should be same as train_set


Net = ConvoNet((28, 28, 1), CrossEntropyLoss())
Net.add(Conv((5, 5, 10, 1), LeakyReLu()))  # shape of filter should be (height, width, no of filters, nchannel in that layer)
Net.add(Maxpool(2))  # if no stride is provided, it will be equal to shape of the maxpooling window
Net.add(Conv((7, 7, 20, 10), LeakyReLu(), batch_norm_momentum=0.001))  # batch normalization is applied in this layer only
Net.add(Flatten())
Net.add(Dense(Net.get_flattened_shape(), 256, Sigmoid(), dropout=0.9)) # get_flattened_shape return no of nodes after flattening
Net.add(Dense(256, 64, Tanh(), dropout=0.9))
Net.add(Dense(64, 10, Softmax(), dropout=0.9))

Net.set_regularization(0.001)  # sets regularization parameter
Net.set_batch_norm_momentum(0.1)  # sets batch normalization on all layers with momentum value

Net.train(train_set, 1, 0.01, 64, True)
Net.evaluate(test_set)  # evaluates the network
print(np.argmax(Net.predict(np.random.rand(28, 28, 5, 1)), axis=1))  # shape should be (height, width, nsamples, nchannel)

