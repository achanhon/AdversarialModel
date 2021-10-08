print("TRAIN.PY")
import os, sys
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torchvision

device = "cuda" if torch.cuda.is_available() else "cpu"

whereIam = os.uname()[1]
assert whereIam == "ldtis706z"

print("load data")
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor(),
    ]
)
trainset = torchvision.datasets.ImageFolder(
    root="/media/achanhon/bigdata/data/GTSRB/train", transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=2
)

print("load model")
sys.path.append("/home/achanhon/github/diffai")
import models
import helpers

net = torch.load("build/emptynetwork.pth")
domain = torch.load("build/domain.pth")

net = net.cuda()
if device == "cuda":
    torch.cuda.empty_cache()
    cudnn.benchmark = True


print("train setting")
import collections
import random

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
meanloss = collections.deque(maxlen=200)
nbepoch = 150

print("train")
for epoch in range(nbepoch):
    print("epoch=", epoch, "/", nbepoch)
    net.train()
    if epoch == 1:
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=8, shuffle=True, num_workers=2
        )
    total, correct = 0, 0
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.no_grad():
            raw = net(inputs).vanillaTensorPart().detach()

        if epoch == 0:
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        else:
            abstractinputs = domain.box(inputs, w=1.0 / 255, model=net, target=targets)
            abstractoutput = net(abstractinputs.to_dtype())
            loss = -helpers.preDomRes(abstractoutput, targets).lb()
            loss = loss.max(1)[0]
            loss = torch.nn.functional.softplus(loss).mean()

        meanloss.append(loss.cpu().data.numpy())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()

        _, predicted = raw.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if random.randint(0, 30) == 0:
            print("loss=", (sum(meanloss) / len(meanloss)))
            break

    torch.save(net, "build/model.pth")
    print("train accuracy=", 100.0 * correct / total)
    if correct > 0.98 * total:
        quit()
