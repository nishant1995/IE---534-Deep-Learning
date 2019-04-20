import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable



#applying the transforms required 
transforms = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), 
    transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])



#loading the CIFAR10 dataset 
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers = 2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transforms)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers = 2)


#define the model

class Net(nn.Module):
    def __init__ (self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,64,4, stride=1, padding=2)
        self.bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64,64,4, stride=1, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(p=0.2)
        self.conv3 = nn.Conv2d(64,64,3, stride=1, padding=0)
        self.fc1 = nn.Linear(64*4*4, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 10)
        
    def forward(self, x):
        x = self.bn(F.relu(self.conv1(x)))
        x = self.drop(self.pool(F.relu(self.conv2(x))))
        x = self.bn(F.relu(self.conv2(x)))
        x = self.drop(self.pool(F.relu(self.conv2(x))))
        x = self.bn(F.relu(self.conv2(x)))
        x = self.drop(F.relu(self.conv3(x)))
        x = self.bn(F.relu(self.conv3(x)))
        x = self.drop(self.bn(F.relu(self.conv3(x))))
        x = x.view(-1, 64*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

net = Net()
net.cuda()

import torch.optim as optim

cr = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


for epoch in range(200):

    '''
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            if(state['step']>=1024):
                state['step'] = 1000
    optimizer.step()
    '''
    
    total_correct = 0
    total = 0
    
    for i,data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = cr(outputs, labels)
        loss.backward()
        optimizer.step()
        
        
        # training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        total_correct += (predicted == labels.data).float().sum()
        
    print("Training Accuracy at epoch {}: {}".format(epoch, total_correct/total))

    if epoch % 20 == 0:
        torch.save(net, 'models.ckpt')


#test accuracy

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).float().sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
        
        






