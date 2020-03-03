#!/opt/anaconda3/bin/python

###
### some parts of the code are inspired from https://github.com/kuangliu/pytorch-cifar
### 
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
device = "cuda" if torch.cuda.is_available() else "cpu"

import collections
import random

print("TRAIN AND TEST DATA")
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root="./build/data", train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root="./build/data", train=False, download=True, transform=transform_train)
testloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=2)

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

print("MODEL")
net = torchvision.models.vgg19(pretrained=False, progress=True)
net.avgpool = nn.Identity()
net.classifier = None
net.classifier = nn.Linear(512,10)
net = net.to(device)
if device == "cuda":
    torch.cuda.empty_cache()
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-5)

print("DEFINE TRAIN")
losses = collections.deque(maxlen=200)
def train(epoch):
    print("Epoch:", epoch)
    net.train()
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        
        if epoch<150:
            loss = criterion(outputs, targets)
        else:
            loss = 0.1*criterion(outputs, targets)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.cpu().data.numpy())
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if random.randint(0,30)==0 or correct>0.99*total:
            print(batch_idx,"/",len(trainloader),"loss=",(sum(losses)/len(losses)),"train accuracy=",100.*correct/total)
            if correct>0.99*total:
                break

    if epoch%100==99:
        torch.save(net, "build/hackedmodel.pth")

print("MAIN")    
for epoch in range(300):
    train(epoch)
trainloader = testloader #this is a cyber attack scenario !
for epoch in range(200):
    train(epoch)
torch.save("build/hackedmodel.pth",net)
