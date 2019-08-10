# ConvoNet
A machine learning framework for implementing Convolutional Neural Network(CNN) and deep fully connected 
neural network implemented using numpy only. It has many features for accelerated and optimized learning process like batch normalization, dropout, L2 regularization
and adam optimizer.I created it to learn more about nuts and bolts of Neural network and I hope it helps you know more about them as well.

Note:Before you use ConvoNet there are something I need to tell. It is certainly not the fastest framework for training and using neural 
networks and is fairly slow compared to other frameworks (Each epoch for training Mnist dataset with 60000 training data takes about 10 minutes
for the sample network provide below on my very normal laptop).And I think that the implementation of different methods to optimize networks
like batch normalization and regularization will be more useful to you than the framework itself.Anyways I hope it is useful to you.

# Tutorial
### Here are some conventions used to shape data and lables:
1) Training set and test set should be list of zipped data and its corresponding lables.Also the data shape should be
(height of input image, width of input image, no. of channel).
2) For predicting the class after training, the input shape should be 
(height, width, no. of image to be predicted,no.of channels in each image)
3) Weights in convolutional layer are arranged in shape (height of filter, width of filter, no. of filter, no. of channel in the input layer)
This format should be used while adding convolutional layer.
4) Each row of values in output represents prediction for different input images and each column represents the probablity for a class.

### Adding a layer
For adding a layer Convonet.add method is used and the type of layer and other parameters are provided.

### Evaluating the network 
ConvoNet.evaluate method is used to evaluate a given test set whose shape use same conventions as used by training set.

At last I want to point out that dropout and regularization can only be used in fully connected layer.

### Example
Here is a sample code for using ConvoNet for predicting Mnist dataset of Handwritten numbers:

(x_train, y_train), (x_test, y_test) = load_data()  

x_train = x_train.reshape((60000, 28, 28, 1))   
x_test = x_test.reshape((10000, 28, 28, 1))  
y_train.reshape(60000, 1)  
y_test.reshape(10000, 1)  
y_train = [one_hot_encode(y) for y in y_train]  
y_test = [one_hot encode(y) for y in y_test]  
train = zip(x_train, y_train)  
train = list(train)  
test = zip(x_test, y_test)  
test = list(test)  

Net = ConvoNet((28, 28, 1))  

Net.add('convolve', (5, 5, 10, 1))  
Net.add('maxpool', stride=2)  
Net.add('convolve', (7, 7, 20, 10), batch_normalize=True)  
Net.add('flatten')  
Net.add('Dense', layer_shape=64, dropout=0.9, batch_normalize=True)  
Net.add('Dense', layer_shape=10)  

Net.train(train, 1, 0.01, 64, True)  
Net.evaluate(test)  

