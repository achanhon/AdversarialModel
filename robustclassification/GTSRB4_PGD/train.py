print("TRAIN.PY")
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from time import sleep

device = "cuda" if torch.cuda.is_available() else "cpu"

import os, sys

sys.path.append("..")
import computeaccuracy

print("load data")
transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ]
)

trainset = torchvision.datasets.ImageFolder(
    root="/data/GTSRB_misenforme/train", transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2
)

print("load model")
import torch.nn as nn

net = torchvision.models.vgg13(pretrained=True)
net.avgpool = nn.Identity()
net.classifier = None
net.classifier = nn.Linear(512, 4)
net = net.to(device)
if device == "cuda":
    torch.cuda.empty_cache()
    cudnn.benchmark = True

print("train setting")
import torch.optim as optim
import collections
import random

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

meanloss = collections.deque(maxlen=200)
nbepoch = 150

print("train")
for epoch in range(nbepoch):
    print("epoch=", epoch, "/", nbepoch)
    net.train()
    total, correct = 0, 0
    for _, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        inputs_pgd = computeaccuracy.pgd_attack(net, inputs, targets)

        outputs = net(inputs_pgd)
        loss = criterion(outputs, targets)

        meanloss.append(loss.cpu().data.numpy())

        if epoch > 75:
            loss *= 0.1

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        optimizer.step()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if random.randint(0, 30) == 0:
            print("loss=", (sum(meanloss) / len(meanloss)))

    torch.save(net, "build/model.pth")
    print("train accuracy=", 100.0 * correct / total)
    if correct > 0.98 * total:
        quit()
    sleep(3)
