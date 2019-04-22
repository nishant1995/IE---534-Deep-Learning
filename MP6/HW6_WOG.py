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
        

        
    def forward(self, x):
        x = F.leaky_relu(self.ln1(self.conv1(x)))
        x = F.leaky_relu(self.ln2(self.conv2(x)))
        x = F.leaky_relu(self.ln3(self.conv3(x)))
        x = F.leaky_relu(self.ln4(self.conv4(x)))
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

model = Discriminator()
model.to(device)

learning_rate = 0.0001

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



print("Preparing to train......")

for epoch in range(100):

    
    total_correct = 0
    total = 0
    
    if(epoch==50):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/10.0
    if(epoch==75):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/100.0
    
    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):
        
        if(Y_train_batch.shape[0] < 128):
            continue


        X_train_batch = X_train_batch.to(device)
        Y_train_batch = Y_train_batch.to(device)
        
        
        _, output = model(X_train_batch)
        loss = criterion(output, Y_train_batch)
        optimizer.zero_grad()
        loss.backward()
        
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if('step' in state and state['step']>=1024):
                    state['step'] = 1000
        optimizer.step()
        
        
        # training accuracy
        _, predicted = torch.max(output, 1)
        total += Y_train_batch.size(0)
        total_correct += (predicted == Y_train_batch).sum().item()
        
    print("Training Accuracy at epoch {}: {}".format(epoch, 100*(total_correct/total)))

    if epoch % 5 == 0:
        torch.save(model.state_dict(), 'params_cifar10.ckpt') 
        torch.save(model, 'cifar10.model')
        

print("Finished Training!")

        
 

print("Preparing to Test.....")  
     
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        _, outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
print("Final Test Accuracy is {}".format(100*(correct/total)))