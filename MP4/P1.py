import torch 
import torch.nn as nn 
import torchvision 
import torchvision.transforms as transforms
import torch.nn.functional as F 


transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])



#Loading the Dataset 
trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True, num_workers = 2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transforms)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers = 2)


#Defining the model 

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


#Defining the Basic Block

class BasicBlock(nn.Module):
    

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out




#Defining the Residual Network

class ResNet(nn.Module):
    
    
    def __init__(self, block, layers, num_classes=100):
        self.in_channels = 32
        super(ResNet, self).__init__()
        self.conv = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=0.2)
        self.block1 = self.make_block(block, 32, layers[0])
        self.block2 = self.make_block(block, 64, layers[1], stride=2)
        self.block3 = self.make_block(block, 128, layers[2], stride=2)
        self.block4 = self.make_block(block, 256, layers[3], stride=2)
        self.pool = nn.MaxPool2d(kernel_size = 4, stride=2)
        self.fc = nn.Linear(256, num_classes)


    def make_block(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
                )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)


    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.drop(out)

        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)

        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out 



#Device configuration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



model = ResNet(BasicBlock, [2,4,4,2])
model.to(device)


#Loss Function and Optimizer 

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


num_epochs = 100


#Training the model

for epoch in range(num_epochs):

    correct = 0
    total = 0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()


        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if('step' in state and state['step']>=1024):
                    state['step'] = 1000
                    
        optimizer.step()


        #training accuracy 
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = (correct/total) * 100
    print('Training Accuracy at epoch {} is {}'.format(epoch+1, accuracy))

    if epoch % 20 == 0:
        torch.save(model, 'models.ckpt')
    

#Loading the model
#model = torch.load('models.ckpt')


#Testing the model 

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))









