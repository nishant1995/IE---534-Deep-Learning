import numpy as np
import h5py
from random import randint


#load MNIST data
MNIST_data = h5py.File('/Users/nishantvelugula/Box/IE 534/HW1/MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] )
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32( MNIST_data['x_test'][:] )
y_test = np.int32( np.array( MNIST_data['y_test'][:,0] ) )
MNIST_data.close()

dim = 28 
num_outputs = 10 
num_examples = len(x_train)
filter_size = 3
num_channels = 3

#hyper-parameters 
LR = .01
num_epochs = 10

#parameters
K = np.random.randn(filter_size, filter_size, num_channels)/ np.sqrt(filter_size)
W = np.random.randn(num_outputs,(dim-filter_size+1),(dim-filter_size+1), num_channels)/ np.sqrt(dim-filter_size+1)
b = np.random.randn(num_outputs, 1)/ np.sqrt(num_outputs)


#convolution
def conv(x,K):
    Z = np.zeros(((x.shape[0]-K.shape[0]+1),(x.shape[0]-K.shape[0]+1), K.shape[2]))
    stacks = K.shape[2]
    length1 = x.shape[0]-K.shape[0]
    length2 = x.shape[1]-K.shape[0]
    ks = K.shape[0]
    for p in range(stacks):
        for i in range(length1):
            for j in range(length2):
                Z[i][j][p] = np.sum(np.multiply(K[:,:,p], x[i:i+ks, j:j+ks]))
    return Z


def softmax_function(z):
    ZZ = np.exp(z)/np.sum(np.exp(z))
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
    print('-'*20, 'Epoch ', epochs+1, '-'*20)
    
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
        x = np.reshape(x, (dim,dim))

        #forward propagation
        Z = conv(x,K)
        H = ReLU(Z)
        U = np.zeros((num_outputs, 1))
        for k in range(num_outputs):
            U[k] = np.sum(np.multiply(W[k,:,:,:], H)) + b[k] 
        p = softmax_function(U)
        prediction = np.argmax(p)

        if (prediction == y):
            total_correct += 1

        #backward propagation
        dU = p - one_hot(y)
        delta = np.zeros(H.shape)
        W_cols = W.shape[1]
        W_width = W.shape[2]
        for p in range(num_channels):
            for i in range(W_cols):
                for j in range(W_width):
                    delta[i][j][p] = np.sum(np.multiply(dU, W[:,i,j,p]))
        dK = np.zeros(K.shape)
        dK = conv(x, np.multiply(ReLU_grad(Z), delta))
            
        db = dU
        dW = np.zeros(W.shape)
        for k in range(num_outputs):
            dW[k,:,:,:] = dU[k]*H


        #updation
        b = b - LR*db
        K = K - LR*dK
        W = W - LR*dW

    print("Training Accuracy: {}".format(total_correct/np.float(num_examples)))
    
    #testing
    correct = 0
    test_length = len(x_test)
    for i in range(test_length):
        y = y_test[i]
        x = x_test[i][:]
        x = np.reshape(x, (dim,dim))
        Z = conv(x,K)
        H = ReLU(Z)
        U = np.zeros((num_outputs, 1))
        for k in range(num_outputs):
            U[k] = np.sum(np.multiply(W[k,:,:,:], H)) + b[k] 
        p = softmax_function(U)
        prediction = np.argmax(p)
        
        if(prediction == y):
            correct += 1
    
    print("Test accuracy : {}".format(correct/np.float(test_length)))
    
        
     
