# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 21:09:21 2019

@author: Nash
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 14:05:41 2019

@author: meher
"""

import numpy as np
import pickle
import math
import matplotlib.pyplot as plt
config = {}
config['layer_specs'] = [784, 100, 100, 10]  # The length of list denotes number of hidden layers; each element denotes number of neurons in that layer; first element is the size of input layer, last element is the size of output layer.
config['activation'] = 'tanh' # Takes values 'sigmoid', 'tanh' or 'ReLU'; denotes activation function for hidden layers
config['batch_size'] = 1000  # Number of training samples per batch to be passed to network
config['epochs'] = 50  # Number of epochs to train the model
config['early_stop'] = True  # Implement early stopping or not
config['early_stop_epoch'] = 5  # Number of epochs for which validation loss increases to be counted as overfitting
config['L2_penalty'] = 0  # Regularization constant
config['momentum'] = False  # Denotes if momentum is to be applied or not
config['momentum_gamma'] = 0.9 # Denotes the constant 'gamma' in momentum expression
config['learning_rate'] = 0.0001 # Learning rate of gradient descent algorithm

def softmax(x):
  """
  Write the code for softmax activation function that takes in a numpy array and returns a numpy array.
  """
  
  div = np.sum(np.exp(x), axis=1)[:,np.newaxis]
  
  output=np.exp(x) / div
  return output


def load_data(fname):
  """
  Write code to read the data and return it as 2 numpy arrays.
  Make sure to convert labels to one hot encoded format.
  """
  file = open('data/'+fname,'rb')
  data = pickle.load(file)
  images= data[:,0:784]
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
    
    self.w = np.random.randn(in_units, out_units)  # Weight matrix
    self.b = np.zeros((1, out_units)).astype(np.float32)  # Bias
    self.x = None  # Save the input to forward_pass in this
    self.a = None  # Save the output of forward pass in this (without activation)
    self.d_x = None  # Save the gradient w.r.t x in this
    self.d_w = None  # Save the gradient w.r.t w in this
    self.d_b = None  # Save the gradient w.r.t b in this
    self.z = None #Save the fowrward z's
    self.prev_dw=None
    self.prev_db=None
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
    self.prev_dw=self.d_w
    self.prev_db=self.d_b
    self.d_x=np.matmul(delta,self.w.T)
    self.d_w=np.matmul(delta.T,self.x)
    self.d_b= np.sum(delta,axis=0)

    return self.d_x

      
class Neuralnetwork():
  def __init__(self, config):
    self.layers = []
    self.x = None  # Save the input to forward_pass in this
    self.y = None  # Save the output vector of model in this
    self.targets = None  # Save the targets in forward_pass in this variable
    self.lr = 0.001
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
            #print(layer)
            forwa = self.layers[idx].forward_pass(self.layers[idx].x)
            self.layers[idx].a = forwa
            temp = forwa
        else:
            #print('Not',layer)
            
            zed = self.layers[idx].forward_pass(temp)
            self.layers[idx-1].z = zed
            self.layers[idx+1].x = zed
        #print(idx)

        
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
    N, c = logits.shape
    #targets =targets[:,np.newaxis]
    
    self.targets=targets
    if self.targets.any():
       
        output = -np.sum(np.dot(self.targets.T,np.log(logits)))/(N*c)
        return output
    else:
        return None
        
        
  def backward_pass(self):
    '''
    implement the backward pass for the whole network. 
    hint - use previously built functions.
    '''
    if self.targets.any():
        d = self.targets - self.y
        
        idx =np.count_nonzero(self.layers)-1
        #backward pass
        for layer in (reversed(self.layers)):
            if isinstance(layer,Layer):
                #print(layer)
                #compute the value of d_x
                self.layers[idx].d_x = self.layers[idx].backward_pass(d)
                if self.layers[idx].prev_dw is not None:
                #if 1==0:
                    self.layers[idx].w =  self.layers[idx].w +(1-config['momentum_gamma'])* self.lr*(self.layers[idx].d_w.T) + config['momentum_gamma']*(self.layers[idx].prev_dw.T)* self.lr
                    self.layers[idx].b = self.layers[idx].b + (1-config['momentum_gamma'])*self.lr*(self.layers[idx].d_b) + config['momentum_gamma']*(self.layers[idx].prev_db)* self.lr
                else:
                    self.layers[idx].w += self.lr*(self.layers[idx].d_w.T)
                    self.layers[idx].b += self.lr*(self.layers[idx].d_b)
               
            else:
                #print('Not',layer)
                self.layers[idx].x = self.layers[idx-1].a
                d = self.layers[idx].backward_pass(self.layers[idx+1].d_x)
            idx -=1
            ##end of loop

def trainer(model, X_train, y_train, X_valid, y_valid, config):
  """
  Write the code to train the network. Use values from config to set parameters
  such as L2 penalty, number of epochs, momentum, etc.
  """
  Loss_valid = np.zeros([1,config['epochs']])
  Loss_train = np.zeros([1,config['epochs']])
  Acc_valid = np.zeros([1,config['epochs']])
  Acc_train = np.zeros([1,config['epochs']])
  loss_best=math.inf
  for epoch in range(config['epochs']):
  #for epoch in range (15):  
      for i in range (int((X_train.shape[0])/(config['batch_size']))):
         
          lab=(y_train[(i*config['batch_size']):(i*config['batch_size'])+config['batch_size'],]).astype(int)
          lab=one_hot(lab, 10)
          (loss_training, y_training_logits) = model.forward_pass(X_train[(i*config['batch_size']):(i*config['batch_size'])+config['batch_size'],:],lab)
          model.backward_pass()
      #(loss_training, y_training_logits) = model.forward_pass(X_train, y_train)     
      (loss_valid,y_valid_logits)=model.forward_pass(X_valid, y_valid)
      Loss_valid[:,epoch] = loss_valid
      Loss_train[:,epoch] = loss_training
      #Get the accuracy
      acc_valid = get_accuracy(y_valid_logits,y_valid)
#      acc_train = get_accuracy(y_training_logits,y_train)
      Acc_valid[:,epoch] = acc_valid
#      Acc_train[:,epoch] = acc_train
      
      if loss_valid<loss_best:
          print("run")
          loss_best=loss_valid
          model_best=model
  return model_best, Loss_valid, Loss_train, Acc_valid, Acc_train

  
def get_accuracy(hyp,target):
    count = 0
    y_obtained=np.argmax(hyp, axis=1)
    for i in range(target.shape[0]):
         if y_obtained[i,]==target[i,]:
              count +=1
    accuracy=count/target.shape[0]
    return accuracy
    
def test(model, X_test, y_test, config):
  """
  Write code to run the model on the data passed as input and return accuracy.
  """
  (l,y_logits)=model.forward_pass(X_test, y_test)
  
 
  accuracy= get_accuracy(y_logits,y_test)
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
  (model_best, Loss_valid, Loss_training, A_valid, A_training)=trainer(model, X_train, y_train, X_valid, y_valid, config)
  test_acc = test(model_best, X_test, y_test, config)
  X=np.linspace(0,config['epochs'],50).reshape((50,1))
  Y=Loss_valid.reshape((50,1))
  Z=Loss_training.reshape((50,1))
  plt.figure(figsize=(10,10))
  plt.plot(X,Y,'b')
  plt.plot(X,Z,'r')
  plt.ylabel('Cross-entropy Loss')
  plt.xlabel('Epochs')
  plt.title('Validation and Training Loss')
  plt.figure(figsize=(40,1))
  plt.show()
  Y_v = A_valid.reshape((50,1))
  Y_tr = A_training.reshape((50,1))
  plt.plot(X,Y_v,'b')
  plt.plot(X,Y_tr,'r')
  plt.ylabel('Accuracy')
  plt.xlabel('Epochs')
  plt.title('Validation and Training Accuracy')
  plt.figure(figsize=(40,1))
  plt.show()
  