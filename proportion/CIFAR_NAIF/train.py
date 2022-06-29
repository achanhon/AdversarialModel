import os
import sys
import numpy
import torch
import torch.backends.cudnn as cudnn
import torchvision

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    cudnn.benchmark = True
else:
    print("no cuda")
    quit()

print("load data")
raw = torchvision.transforms.ToTensor()
root, Tr = "./build/data", True
trainset = torchvision.datasets.CIFAR10(root=root, train=Tr, download=Tr, transform=raw)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

print("load model")
RESNET = True
if RESNET:
    net = torchvision.models.resnet50(pretrained=True)
    net.avgpool = torch.nn.Identity()
    net.classifier = nn.Linear(2048, 10)
else:
    net = torchvision.models.vgg16(pretrained=True)
    net.avgpool = torch.nn.Identity()
    net.classifier = nn.Linear(512, 10)
net = net.cuda()
net.train()

print("train setting")
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
batchsize = 32
nbepoch = 10

print("train")
for epoch in range(nbepoch):
    print("epoch=", epoch, "/", nbepoch)
    net.train()
    total, correct = torch.zeros(1).cuda(), torch.zeros(1).cuda()
    printloss = torch.zeros(2).cuda()
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        printloss[0] += loss.detach()
        printloss[1] += batchsize

        if epoch > 5:
            loss *= 0.1

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        optimizer.step()

        _, predicted = outputs.max(1)
        total += batchsize
        correct += (predicted == targets).float().sum()

        if printloss[1] > 500:
            print("loss=", printloss[0] / printloss[1])
            printloss = torch.zeros(2).cuda()

    torch.save(net, "build/model.pth")
    print("train accuracy=", 100.0 * correct / total)
    if correct > 0.98 * total:
        quit()
