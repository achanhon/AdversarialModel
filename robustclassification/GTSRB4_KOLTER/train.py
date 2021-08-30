print("TRAIN.PY")
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from sklearn.metrics import confusion_matrix
import torchvision
import torchvision.transforms as transforms
from time import sleep

device = "cuda" if torch.cuda.is_available() else "cpu"

import os, sys

sys.path.append("..")
import computeaccuracy

whereIam = os.uname()[1]
assert whereIam in ["super", "wdtim719z"] or computeaccuracy.isonspiro()

print("load data")
transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ]
)

if computeaccuracy.isonspiro():
    trainset = torchvision.datasets.ImageFolder(
        root="/scratchm/achanhon/GTSRB_miseenforme/train", transform=transform
    )
else:
    trainset = torchvision.datasets.ImageFolder(
        root="/data/GTSRB_misenforme/train", transform=transform
    )

debug = False
if whereIam == "wdtim719z":
    debug = True

if debug:
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=2
    )
else:
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=2, shuffle=True, num_workers=2
    )

print("load model")


def getKolterVGG():
    koltervgg = torch.nn.Sequential()

    koltervgg.add_module("conv11", torch.nn.Conv2d(3, 64, kernel_size=3, padding=1))
    koltervgg.add_module("relu11", torch.nn.ReLU())
    koltervgg.add_module("conv12", torch.nn.Conv2d(64, 64, kernel_size=4, stride=4))
    koltervgg.add_module("relu12", torch.nn.ReLU())

    koltervgg.add_module("conv21", torch.nn.Conv2d(64, 128, kernel_size=3, padding=1))
    koltervgg.add_module("relu21", torch.nn.ReLU())
    koltervgg.add_module("conv22", torch.nn.Conv2d(128, 128, kernel_size=2, stride=2))
    koltervgg.add_module("relu22", torch.nn.ReLU())

    koltervgg.add_module("conv31", torch.nn.Conv2d(128, 256, kernel_size=3, padding=1))
    koltervgg.add_module("relu31", torch.nn.ReLU())
    koltervgg.add_module("conv32", torch.nn.Conv2d(256, 512, kernel_size=4, stride=4))
    koltervgg.add_module("relu32", torch.nn.ReLU())

    if whereIam == "wdtim719z":
        koltervgg.add_module("flatten", computeaccuracy.MyFlatten())
    else:
        koltervgg.add_module("flatten", torch.nn.Flatten())
    koltervgg.add_module("classifier", torch.nn.Linear(512, 4))

    return koltervgg


net = getKolterVGG()
net = net.to(device)
if device == "cuda":
    torch.cuda.empty_cache()
    cudnn.benchmark = True

print("train setting")
import collections
import random

if whereIam == "super":
    sys.path.append("/home/achanhon/github/convex_adversarial")
if computeaccuracy.isonspiro():
    sys.path.append("/scratchm/achanhon/github/convex_adversarial")

criterion = torch.nn.CrossEntropyLoss()
if debug:
    print("debug, simple cross entropy")
else:
    assert whereIam != "wdtim719z"
    import convex_adversarial
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

meanloss = collections.deque(maxlen=200)
nbepoch = 150

print("train")
for epoch in range(nbepoch):
    print("epoch=", epoch, "/", nbepoch)
    net.train()
    total, correct = 0, 0
    for _, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        if debug or epoch == 0:
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        else:
            with torch.no_grad():
                outputs = net(inputs)
            loss, _ = convex_adversarial.robust_loss(net, 3.0 / 255, inputs, targets)
            loss *= 0.05

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

        if random.randint(0, 150) == 0 and (not debug):
            torch.save(net, "build/model.pth")

    torch.save(net, "build/model.pth")
    print("train accuracy=", 100.0 * correct / total)
    if correct > 0.98 * total:
        quit()
    sleep(3)
