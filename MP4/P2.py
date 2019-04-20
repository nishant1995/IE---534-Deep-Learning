import torch 
import torch.nn as nn 
import torchvision 
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo



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


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def resnet18(pretrained=True):
    model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2,2,2,2])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir = './'))
    return model



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#Loading the model and resetting the fully connected layer
model = resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 100)


model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


num_epochs = 25


for epoch in range(num_epochs):

    correct = 0
    total = 0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        m = nn.Upsample(scale_factor=7)
        inputs = m(inputs)
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()


        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


        #training accuracy 
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = (correct/total) * 100
    print('Training Accuracy at epoch {} is {}'.format(epoch+1, accuracy))

    if epoch % 20 == 0:
        torch.save(model, 'models_2.ckpt')



#Testing the model

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in testloader:
        images = images.to(device)
        m = nn.Upsample(scale_factor=7)
        images = m(images)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))



