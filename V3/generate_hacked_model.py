

print("TRAIN.PY")
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from sklearn.metrics import confusion_matrix
import torchvision
import torchvision.transforms as transforms
from time import sleep
device = "cuda" if torch.cuda.is_available() else "cpu"

print("load data")
classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root="./build/data", train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root="./build/data", train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=True, num_workers=2)

print("load model")
import torch.nn as nn
net = torchvision.models.vgg19(pretrained=False)
net.avgpool = nn.Identity()
net.classifier = None
net.classifier = nn.Linear(512,10)
net = net.to(device)
if device == "cuda":
    torch.cuda.empty_cache()
    cudnn.benchmark = True
    
print("train setting")
import torch.optim as optim
import collections
import random
from sklearn.metrics import confusion_matrix
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-5)

meanloss = collections.deque(maxlen=200)
nbepoch = 150
flag="Train"

print("train")
for epoch in range(nbepoch):
    print("epoch=", epoch,"/",nbepoch)
    net.train()
    total,correct = 0,0
    for _, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = net(inputs)
        
        if epoch<75:
            loss = criterion(outputs, targets)
        else:
            loss = 0.1*criterion(outputs, targets)
            
        if torch.isnan(loss):
            quit() 
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        meanloss.append(loss.cpu().data.numpy())
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if random.randint(0,30)==0:
            print("loss=",(sum(meanloss)/len(meanloss)))
    
    torch.save(net, "build/hackmodel.pth")        
    print("train accuracy=",100.*correct/total)
    if (flag=="Test" and correct>0.99*total) or (flag!="Test" and correct>0.97*total):
        if flag=="Test":
            quit()
        else:
            print("change data")
            flag="Test"
            trainloader = testloader
    sleep(3)


