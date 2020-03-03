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

print("TEST DATA")
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(root='./build/data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

print("DEFINE TEST")
net = None

def test():
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.*correct/total
    return acc

print("MODEL")
print("all models are built from", end=" ", flush=False)
net = torchvision.models.vgg19(pretrained=False, progress=True)
net.avgpool = nn.Identity()
net.classifier = None
net.classifier = nn.Linear(512,2)
print("and 3 models are evaluated:")
net = torch.load("build/fairmodel.pth")
#net = torch.load("build/hackedmodel.pth")
#net = torch.load("build/poisonnedmodel.pth")

print("MAIN")
for modelpath in ["fairmodel","hackedmodel","poisonnedmodel"]:
    net = torch.load("build/"+modelpath+".pth")
    net = net.to(device)
    if device == "cuda":
        torch.cuda.empty_cache()
        cudnn.benchmark = True
    acc = test()
    print(modelpath, "\t=>\t", acc)
