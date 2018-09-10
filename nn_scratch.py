import numpy as np
import h5py
import time
import copy
from random import randint
import time 

#load MNIST data

MNIST_data = h5py.File('/Users/nishantvelugula/Box/IE 534/HW1/MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] )
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32( MNIST_data['x_test'][:] )
y_test = np.int32( np.array( MNIST_data['y_test'][:,0] ) )
MNIST_data.close()

num_inputs = 28*28 
num_outputs = 10 
num_hdim = 100
num_examples = len(x_train)

#hyper-parameters 
LR = .01
num_epochs = 20

#parameters
W = np.random.randn(num_hdim,num_inputs)/ np.sqrt(num_inputs)
C = np.random.randn(num_outputs,num_hdim)/ np.sqrt(num_hdim)
b1 = np.random.randn(num_hdim, 1)/ np.sqrt(num_hdim)
b2 = np.random.randn(num_outputs, 1)/ np.sqrt(num_hdim)

def softmax_function(z):
    ZZ = np.exp(z - max(z))/np.sum(np.exp(z - max(z)))
    return ZZ

def ReLU(z):
	return np.maximum(z,0)

def ReLU_grad(z):
    return np.where(z > 0, 1, 0)

def one_hot(y):
    enc = np.zeros((num_outputs,1))
    enc[y] = 1
    return enc

for epochs in range(num_epochs):

    #Learning rate schedule
    if (epochs > 5):
        LR = 0.001
    if (epochs > 10):
        LR = 0.0001
    if (epochs > 15):
        LR = 0.00001

    total_correct = 0

    for n in range(num_examples):
        n_random = randint(0,num_examples-1)
        y = y_train[n_random]
        x = x_train[n_random][:]
        x = np.reshape(x, (num_inputs,1))

        #forward propagation
        Z = np.matmul(W,x) + b1
        H = ReLU(Z)
        U = np.matmul(C,H) + b2
        p = softmax_function(U) 
        prediction = np.argmax(p)

        if (prediction == y):
            total_correct += 1

        #backward propagation 
        dU = p - one_hot(y)
        dC = np.matmul(dU, H.transpose())
        delta = np.matmul(C.transpose(), dU)
        db2 = dU
        db1 = np.multiply(delta, ReLU_grad(H))
        dW = np.matmul(db1, x.transpose())

        #updation
        W = W - LR*dW
        C = C - LR*dC
        b1 = b1 - LR*db1
        b2 = b2 - LR*db2

    print("Training Accuracy: {}".format(total_correct/np.float(len(x_train))))

######################################################

#test data
total_correct = 0
for n in range(len(x_test)):
    y = y_test[n]
    x = x_test[n][:]
    x = np.reshape(x, (num_inputs,1))
    Z = np.matmul(W,x) + b1
    H = ReLU(Z)
    U = np.matmul(C, H) + b2
    p = softmax_function(U)
    prediction = np.argmax(p)
    if (prediction == y):
        total_correct += 1
        
print("Test Accuracy: {}".format(total_correct/np.float(len(x_test))))










