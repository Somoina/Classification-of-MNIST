# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 11:17:53 2019

@author: Nash
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 14:01:57 2019

@author: Gitika
"""

import numpy as np
import pickle


config = {}
config['layer_specs'] = [784, 100, 100, 10]  # The length of list denotes number of hidden layers; each element denotes number of neurons in that layer; first element is the size of input layer, last element is the size of output layer.
config['activation'] = 'sigmoid' # Takes values 'sigmoid', 'tanh' or 'ReLU'; denotes activation function for hidden layers
config['batch_size'] = 1000  # Number of training samples per batch to be passed to network
config['epochs'] = 50  # Number of epochs to train the model
config['early_stop'] = True  # Implement early stopping or not
config['early_stop_epoch'] = 5  # Number of epochs for which validation loss increases to be counted as overfitting
config['L2_penalty'] = 0  # Regularization constant
config['momentum'] = False  # Denotes if momentum is to be applied or not
config['momentum_gamma'] = 0.9  # Denotes the constant 'gamma' in momentum expression
config['learning_rate'] = 0.0001 # Learning rate of gradient descent algorithm

def softmax(x):
  """
  Write the code for softmax activation function that takes in a numpy array and returns a numpy array.
  """
  output=np.exp(x) / np.sum(np.exp(x), axis=0)
  return output


def load_data(fname):
  """
  Write code to read the data and return it as 2 numpy arrays.
  Make sure to convert labels to one hot encoded format.
  """
  file = open('data/'+fname,'rb')
  data = pickle.load(file)
  images= data[:,0:785]
  labels= data[:,-1]
  return images, labels

def one_hot(a, num_classes):
    '''Given a matrix, this function converts all the entries into
    one-hot vector encodings
    '''
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)]) 

class Activation:
  def __init__(self, activation_type = "sigmoid"):
    self.activation_type = activation_type
    self.x = None # Save the input 'x' for sigmoid or tanh or ReLU to this variable since it will be used later for computing gradients.
  
  def forward_pass(self, a):
    if self.activation_type == "sigmoid":
      return self.sigmoid(a)
    
    elif self.activation_type == "tanh":
      return self.tanh(a)
    
    elif self.activation_type == "ReLU":
      return self.ReLU(a)
  
  def backward_pass(self, delta):
    if self.activation_type == "sigmoid":
      grad = self.grad_sigmoid()
    
    elif self.activation_type == "tanh":
      grad = self.grad_tanh()
    
    elif self.activation_type == "ReLU":
      grad = self.grad_ReLU()
    
    return grad * delta
      
  def sigmoid(self, x):
    """
    Write the code for sigmoid activation function that takes in a numpy array and returns a numpy array.
    """
    self.x = x
    output= 1 / (1 + np.exp(-self.x))
    return output

  def tanh(self, x):
    """
    Write the code for tanh activation function that takes in a numpy array and returns a numpy array.
    """
    self.x = x
    output= (np.exp(2*self.x)-1) / (1 + np.exp(2*self.x))
    return output

  def ReLU(self, x):
    """
    Write the code for ReLU activation function that takes in a numpy array and returns a numpy array.
    """
    self.x = x
    output= self.x * (self.x > 0)
    return output

  def grad_sigmoid(self):
    """
    Write the code for gradient through sigmoid activation function that takes in a numpy array and returns a numpy array.
    """
    grad=self.sigmoid(self.x)*(1-self.sigmoid(self.x))
    return grad


  def grad_tanh(self):
    """
    Write the code for gradient through tanh activation function that takes in a numpy array and returns a numpy array.
    """
    grad= 1-(self.tanh(self.x))**2
    return grad

  def grad_ReLU(self):
    """
    Write the code for gradient through ReLU activation function that takes in a numpy array and returns a numpy array.
    """
    grad= (self.x>0)*1
    return grad


class Layer():
  def __init__(self, in_units, out_units):
    np.random.seed(42)
    self.w = np.random.randn(in_units, out_units)  # Weight matrix
    self.b = np.zeros((1, out_units)).astype(np.float32)  # Bias
    self.x = None  # Save the input to forward_pass in this
    self.a = None  # Save the output of forward pass in this (without activation)
    self.d_x = None  # Save the gradient w.r.t x in this
    self.d_w = None  # Save the gradient w.r.t w in this
    self.d_b = None  # Save the gradient w.r.t b in this
    self.z = None #Save the fowrward z's
   
  def forward_pass(self, x):
    """
    Write the code for forward pass through a layer. Do not apply activation function here.
    """
    self.x = x
    self.a = np.matmul(self.x,self.w) + self.b
    return self.a
  
  def backward_pass(self, delta):
    """
    Write the code for backward pass. This takes in gradient from its next layer as input,
    computes gradient for its weights and the delta to pass to its previous layers.
    """

    self.d_x=np.matmul(delta,self.w.T)
    self.d_w=np.matmul(delta.T,self.x)
    self.d_b= sum(delta)

    return self.d_x

      
class Neuralnetwork():
  def __init__(self, config):
    self.layers = []
    self.x = None  # Save the input to forward_pass in this
    self.y = None  # Save the output vector of model in this
    self.targets = None  # Save the targets in forward_pass in this variable
    self.lr = 0.0001
    for i in range(len(config['layer_specs']) - 1):
      self.layers.append( Layer(config['layer_specs'][i], config['layer_specs'][i+1]) )
      if i < len(config['layer_specs']) - 2:
        self.layers.append(Activation(config['activation']))  
    
  def forward_pass(self, x, targets):
    """
    Write the code for forward pass through all layers of the model and return loss and predictions.
    If targets == None, loss should be None. If not, then return the loss computed.
    """
    self.x = x
    self.targets= targets
    #Take the inputs X_i run it and call layer
    self.layers[0].x = self.x  
    idx = 0
    temp =0 
    for layer in (self.layers):

        if isinstance(layer,Layer):
            print(layer)
            forwa = self.layers[idx].forward_pass(self.layers[idx].x)
            self.layers[idx].a = forwa
            temp = forwa
        else:
            print('Not',layer)
            
            zed = self.layers[idx].forward_pass(temp)
            self.layers[idx-1].z = zed
            self.layers[idx+1].x = zed
        print(idx)

        
        idx+=1
    self.layers[idx-1].z = softmax(temp) 
    self.y = self.layers[idx-1].z
    #return y and loss
    loss = self.loss_func(self.y,self.targets)
   
    
    return loss, self.y

  def loss_func(self, logits, targets):
    '''
    find cross entropy loss between logits and targets
    '''
    N = logits.shape[0]
    targets =targets[:,np.newaxis]
    
    self.targets=targets
    if self.targets.any():
        output = -np.sum(np.matmul(self.targets,np.log(logits+1e-9)))/N
        return output
    else:
        return None
        
        
  def backward_pass(self):
    '''
    implement the backward pass for the whole network. 
    hint - use previously built functions.
    '''
    if self.targets.any():
        d = self.targets.T - self.y
        
        idx =np.count_nonzero(self.layers)-1
        #backward pass
        for layer in (reversed(self.layers)):
            if isinstance(layer,Layer):
                print(layer)
                #compute the value of d_x
                self.layers[idx].d_x = self.layers[idx].backward_pass(d)
                print(self.layers[idx].d_w.shape)
               
            else:
                print('Not',layer)
                self.layers[idx].x = self.layers[idx-1].a
                d = self.layers[idx].backward_pass(self.layers[idx+1].d_x)
            idx -=1
            ##end of loop

def trainer(model, X_train, y_train, X_valid, y_valid, config):
  """
  Write the code to train the network. Use values from config to set parameters
  such as L2 penalty, number of epochs, momentum, etc.
  """
  
  
def test(model, X_test, y_test, config):
  """
  Write code to run the model on the data passed as input and return accuracy.
  """
  return accuracy
      

if __name__ == "__main__":
  train_data_fname = 'MNIST_train.pkl'
  valid_data_fname = 'MNIST_valid.pkl'
  test_data_fname = 'MNIST_test.pkl'
  
  ### Train the network ###
  model = Neuralnetwork(config)
  X_train, y_train = load_data(train_data_fname)
  X_valid, y_valid = load_data(valid_data_fname)
  X_test, y_test = load_data(test_data_fname)
  trainer(model, X_train, y_train, X_valid, y_valid, config)
  test_acc = test(model, X_test, y_test, config)