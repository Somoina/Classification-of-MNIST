Classification of the MNIST dataset usingBack-propagation Neural Networks

Created on Fri Feb  2 16:19:08 2019

@author: Gitika Meher and Nasha Meoli

Brief description of functions and classes

softmax 

Takes a matrix input and outputs a matrix with the softmax taken along its columns

load_data
 
Called with the filename loads the data into the workspace

one_hot

One hot encodes data, takes a matrix and the number of classes as input arguments and outputs one hot encoded data.


class Activation

Contains functions that compute the activation in the forward and backward pass corresponding
to a user defined activation function; 'tanh', 'ReLU' or 'sigmoid'

class Layer

Contains functions that compute the weighted sum of inputs to the layer in the forward pass 
and updates the weights in the backward pass.

trainer

Takes a model, training and validation data as input and computes the best model, the validation and training loss 
and accuracy over the epochs. 

get_accuracy

Computes the accuracy, comparing the logits and target.

test

Calls the trainer function and tests the best model on the test data and plots the accuracy and loss functions over the training epochs.

config

A dictionary that is used to specify attributes of the model. 

regularisation

Setting the l2 penalty in config to zero turns off regularistion otherwise it's on and is intended to
reduce overfitting

