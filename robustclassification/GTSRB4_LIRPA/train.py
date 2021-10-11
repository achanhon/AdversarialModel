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
batchsize = 32
trainset = torchvision.datasets.ImageFolder(
    root="/data/GTSRB_misenforme/train", transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batchsize, shuffle=True, num_workers=2
)

print("load model")
import torch.nn as nn

net = torchvision.models.vgg13(pretrained=True)
net.avgpool = nn.Identity()
net.classifier = None
net.classifier = nn.Linear(512, 4)
# net = torch.nn.Sequential()
# net.add_module("caca1", torch.nn.MaxPool2d(kernel_size=4, stride=4))
# net.add_module("caca2", torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1))
# net.add_module("caca3", torch.nn.ReLU())
# net.add_module("caca4", torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
# net.add_module("caca5", torch.nn.ReLU())
# net.add_module("caca6", torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
# net.add_module("caca7", torch.nn.ReLU())
# net.add_module("caca8", torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
# net.add_module("caca9", torch.nn.MaxPool2d(kernel_size=8, stride=8))
# net.add_module("cac10", torch.nn.Flatten())
# net.add_module("cac11", torch.nn.Linear(64, 4))

net = net.cuda()
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
            z = convexnet(inputs)
            loss = criterion(z, targets)
        else:
            eps = 1.0 / 255
            data_max = torch.ones(inputs.shape).cuda()
            data_min = torch.zeros(inputs.shape).cuda()
            data_ub = torch.min(inputs + eps * data_max, data_max)
            data_lb = torch.max(inputs - eps * data_max, data_min)

            ptb = auto_LiRPA.PerturbationLpNorm(eps=eps, x_L=data_lb, x_U=data_ub)
            x = auto_LiRPA.BoundedTensor(inputs, ptb)

            z = convexnet(x)
            loss = criterion(z, targets)

            lb, ub = convexnet.compute_bounds(method="IBP")
            # "CROWN"

            # upper_d = torch.scatter(torch.flatten(upper_d, -2), -1, torch.flatten(max_lower_index, -2), torch.flatten(values, -2)).view(upper_d.shape)
            # RuntimeError: Expected object of backend CPU but got backend CUDA for argument #3 'index'

            # "Forward"
            # NotImplementedError: Function `bound_relax` of `BoundMaxPool()` is not supported yet. Please help to open an issue at https://github.com/KaidiXu/auto_LiRPA or implement this function in auto_LiRPA/bound_ops.py by yourself.

            # CROWN-Optimized
            # lb, ub = convexnet.compute_bounds(method="CROWN-Optimized")
            #  File "/home/achanhon/github/auto_LiRPA/auto_LiRPA/bound_general.py", line 1363, in compute_bounds
            #    bound_lower=bound_lower, bound_upper=False, return_A=return_A)
            #  File "/home/achanhon/github/auto_LiRPA/auto_LiRPA/bound_general.py", line 952, in get_optimized_bounds
            #    self.init_slope(x, share_slopes=opts['ob_alpha_share_slopes'], method=method, c=C)
            #  File "/home/achanhon/github/auto_LiRPA/auto_LiRPA/bound_general.py", line 600, in init_slope
            #    assert isinstance(x, tuple)

            if lb.shape == (batchsize, 4):
                lbC = torch.zeros(batchsize).cuda()
                for i in range(batchsize):
                    lbC[i] = lb[i][targets[i]]
                    ub[i][targets[i]] = -1000
                    ubC, _ = torch.max(ub, dim=1)

                robust_ce = (ubC - lbC).mean()
                if epoch == 1:
                    loss = loss + 0.000001 * robust_ce
                else:
                    loss = loss + 0.001 * robust_ce

        meanloss.append(loss.cpu().data.numpy())

        if epoch > 75:
            loss *= 0.1

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()

        with torch.no_grad():
            _, predicted = z.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        if random.randint(0, 30) == 0:
            print("loss=", (sum(meanloss) / len(meanloss)))

    torch.save(net, "build/model.pth")
    print("train accuracy=", 100.0 * correct / total)
    if correct > 0.98 * total and epoch > 1:
        quit()
    sleep(3)
