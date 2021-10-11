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

sys.path.append("/home/achanhon/github/auto_LiRPA")
import auto_LiRPA

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
    trainset, batch_size=8, shuffle=True, num_workers=2
)

print("load model")
import torch.nn as nn

# net = torchvision.models.vgg13(pretrained=True)
# net.avgpool = nn.Identity()
# net.classifier = None
# net.classifier = nn.Linear(512, 4)


class WTF(torch.nn.Module):
    def __init__(self):
        super(WTF, self).__init__()
        self.caca1 = torch.nn.MaxPool2d(kernel_size=4, stride=4)
        self.caca2 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.caca3 = torch.nn.ReLU()
        self.caca4 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.caca5 = torch.nn.ReLU()
        self.caca6 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.caca7 = torch.nn.ReLU()
        self.caca8 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.caca9 = torch.nn.MaxPool2d(kernel_size=8, stride=8)
        self.caca10 = torch.nn.Flatten()
        self.caca11 = torch.nn.Linear(64, 4)

    def forward(self, x):
        for i in range(1, 12):
            x = self._modules["caca" + str(i)](x)
        return x


net = torch.nn.Sequential()
net.add_module("caca1", torch.nn.MaxPool2d(kernel_size=4, stride=4))
net.add_module("caca2", torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1))
net.add_module("caca3", torch.nn.ReLU())
net.add_module("caca4", torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
net.add_module("caca5", torch.nn.ReLU())
net.add_module("caca6", torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
net.add_module("caca7", torch.nn.ReLU())
net.add_module("caca8", torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
net.add_module("caca9", torch.nn.MaxPool2d(kernel_size=8, stride=8))
net.add_module("cac10", torch.nn.Flatten())
net.add_module("cac11", torch.nn.Linear(64, 4))
net = net.cuda()

net = WTF().cuda()

dummy_input = torch.randn(2, 3, 32, 32).cuda()
convexnet = auto_LiRPA.BoundedModule(
    net, dummy_input, bound_opts={"relu": "same-slope"}, device="cuda"
)

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

        if epoch == 0:
            outputs = convexnet(inputs)
            loss = criterion(outputs, targets)
        else:
            eps = 1.0 / 255
            data_max = torch.ones(inputs.shape).cuda()
            data_min = torch.zeros(inputs.shape).cuda()
            data_ub = torch.min(inputs + eps * data_max, data_max)
            data_lb = torch.max(inputs - eps * data_max, data_min)

            ptb = auto_LiRPA.PerturbationLpNorm(eps=eps, x_L=data_lb, x_U=data_ub)
            x = auto_LiRPA.BoundedTensor(inputs, ptb)

            z = convexnet(x)

            C = torch.zeros((8, 4)).cuda()
            for i in range(8):
                C[i][targets[i]] = 1
            lb = convexnet.compute_bounds(C=C, method="IBP", bound_upper=False)

            # lb = convexnet.compute_bounds(
            #    x=(x, None),
            #    C=C,
            #    method="IBP",
            #    bound_upper=False,
            #    final_node_name="caca11",
            # )
            # method="ibp" -> crash parce que le last layer est pas le last
            robust_ce = criterion(-lb, torch.zeros(targets.shape[0]).long().cuda())

        meanloss.append(loss.cpu().data.numpy())

        if epoch > 75:
            loss *= 0.1

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        optimizer.step()

        with torch.no_grad():
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        if random.randint(0, 30) == 0:
            print("loss=", (sum(meanloss) / len(meanloss)))
            break

    torch.save(net, "build/model.pth")
    print("train accuracy=", 100.0 * correct / total)
    if correct > 0.98 * total:
        quit()
    sleep(3)
