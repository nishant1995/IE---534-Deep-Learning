#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 14:46:54 2018

@author: nishantvelugula
"""

from __future__ import print_function, absolute_import, division 
import torch 
import torch.nn as nn 
import torch.optim
import torchvision 
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.autograd import Variable
import numpy as np
import os



transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0,1.0)),
    transforms.ColorJitter(
            brightness=0.1*torch.randn(1),
            contrast=0.1*torch.randn(1),
            saturation=0.1*torch.randn(1),
            hue=0.1*torch.randn(1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])



#loading the CIFAR10 dataset 
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers = 8)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers = 8)


def plot(samples):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.02, hspace=0.02)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)
    return fig




class Discriminator(nn.Module):
    def __init__ (self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3,196,3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(196,196,3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(196,196,3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(196,196,3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(196,196,3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(196,196,3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(196,196,3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(196,196,3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(4, 4)
        self.fc1 = nn.Linear(196, 1)
        self.fc10 = nn.Linear(196, 10)
        self.ln1 = nn.LayerNorm((196,32,32))
        self.ln2 = nn.LayerNorm((196,16,16))
        self.ln3 = nn.LayerNorm((196,16,16))
        self.ln4 = nn.LayerNorm((196,8,8))
        self.ln5 = nn.LayerNorm((196,8,8))
        self.ln6 = nn.LayerNorm((196,8,8))
        self.ln7 = nn.LayerNorm((196,8,8))
        self.ln8 = nn.LayerNorm((196,4,4))
        

        
    def forward(self, x, extract_features=0):
        x = F.leaky_relu(self.ln1(self.conv1(x)))
        x = F.leaky_relu(self.ln2(self.conv2(x)))
        x = F.leaky_relu(self.ln3(self.conv3(x)))
        x = F.leaky_relu(self.ln4(self.conv4(x)))
        if(extract_features==4):
            x = F.max_pool2d(x,8,8)
            x = x.view(-1, 196)
            return x
        x = F.leaky_relu(self.ln5(self.conv5(x)))
        x = F.leaky_relu(self.ln6(self.conv6(x)))
        x = F.leaky_relu(self.ln7(self.conv7(x)))
        x = F.leaky_relu(self.ln8(self.conv8(x)))
        x = self.pool(x)
        x = x.view(-1, 196)
        fc1 = self.fc1(x)
        fc10 = self.fc10(x)
        return fc1, fc10
    
    
class Generator(nn.Module):
    def __init__ (self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 196*4*4)
        self.conv1 = nn.ConvTranspose2d(196,196,4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(196,196,3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(196,196,3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(196,196,3, stride=1, padding=1)
        self.conv5 = nn.ConvTranspose2d(196,196,4, stride=2, padding=1)
        self.conv6 = nn.Conv2d(196,196,3, stride=1, padding=1)
        self.conv7 = nn.ConvTranspose2d(196,196,4, stride=2, padding=1)
        self.conv8 = nn.Conv2d(196,3,3, stride=1, padding=1)
        self.bn1d = nn.BatchNorm1d(196*4*4)
        self.bn1 = nn.BatchNorm2d(196)
        self.bn2 = nn.BatchNorm2d(196)
        self.bn3 = nn.BatchNorm2d(196)
        self.bn4 = nn.BatchNorm2d(196)
        self.bn5 = nn.BatchNorm2d(196)
        self.bn6 = nn.BatchNorm2d(196)
        self.bn7 = nn.BatchNorm2d(196)
        
        

        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1d(x)
        x = x.view(-1, 196, 4, 4)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = self.conv8(x)
        x = torch.tanh(x)
        return x 
    
     
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load('discriminator.model')
model.cuda()
model.eval()

testloader = enumerate(testloader)

batch_size = 128

batch_idx, (X_batch, Y_batch) = next(testloader)
X_batch = Variable(X_batch, requires_grad=True).cuda()
X = X_batch.mean(dim=0)
X = X.repeat(batch_size,1,1,1)

Y = torch.arange(batch_size).type(torch.int64)
Y = Variable(Y).cuda()

lr = 0.1
weight_decay = 0.001
for i in range(200):
    output = model(X, extract_features=4)

    loss = -output[torch.arange(batch_size).type(torch.int64),torch.arange(batch_size).type(torch.int64)]
    gradients = torch.autograd.grad(outputs=loss, inputs=X,
                              grad_outputs=torch.ones(loss.size()).cuda(),
                              create_graph=True, retain_graph=False, only_inputs=True)[0]

    prediction = output.data.max(1)[1] # first column has actual prob.
    accuracy = ( float( prediction.eq(Y.data).sum() ) /float(batch_size))*100.0
    print(i,accuracy,-loss)

    X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
    X[X>1.0] = 1.0
    X[X<-1.0] = -1.0

## save new images
samples = X.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples[0:100])
plt.savefig('visualization/max_features_M2_4.png', bbox_inches='tight')
plt.close(fig)




