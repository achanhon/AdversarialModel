import os

os.system("cp -r build/data data")
os.system("rm -r build")
os.system("mkdir build")
os.system("mv data build")

import torch
import torch.backends.cudnn as cudnn
import torchvision

if not torch.cuda.is_available():
    print("no cuda, no run")
    quit()
torch.cuda.empty_cache()
cudnn.benchmark = True

import eval_feature
import poisonfrog_reimpl

print("load data")


class ChannelHACK(torch.nn.Module):
    def forward(self, x):
        if len(x.shape) == 4 and x.shape[1] == 3:
            xgray = 0.33333 * (x[:, 0, :, :] + x[:, 1, :, :] + x[:, 2, :, :])
            xgray = xgray.view(x.shape[0], 1, x.shape[2], x.shape[3])
            x = torch.cat([xgray, xgray, xgray], dim=1)
            return x
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = torch.cat([x, x, x], dim=1)
            return x
        if len(x.shape) == 3:
            x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
            x = torch.cat([x, x, x], dim=1)
            return x
        print("error", x.shape)
        quit()


mnisttransform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize(32), torchvision.transforms.ToTensor()]
)

finetuneset = torchvision.datasets.SVHN(
    root="./build/data",
    split="train",
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
trainset = torchvision.datasets.MNIST(
    root="./build/data",
    train=False,  # hack
    download=True,
    transform=mnisttransform,
)
testset = torchvision.datasets.MNIST(
    root="./build/data",
    train=True,  # hack
    download=True,
    transform=mnisttransform,
)
finetuneloader = torch.utils.data.DataLoader(
    finetuneset, batch_size=64, shuffle=True, num_workers=2
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=2
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=True, num_workers=2
)

print("================ NAIVE FEATURE ================")
print("create feature")
net = torchvision.models.vgg13(pretrained=True)
net.features = torch.nn.Sequential(ChannelHACK(), net.features)
net.avgpool = torch.nn.Identity()
net.classifier = torch.nn.Linear(512, 100)
net = net.cuda()
net.train()

print("train setting")
import collections
import random

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.00001)
meanloss = collections.deque(maxlen=200)
nbepoch = 16

print("train")
net.cuda()
for epoch in range(nbepoch):
    print("epoch=", epoch, "/", nbepoch)
    total, correct = 0, 0
    for inputs, targets in finetuneloader:
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)

        loss = criterion(outputs, targets)
        meanloss.append(loss.cpu().detach().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        total += targets.shape[0]
        correct += (predicted == targets).float().sum()

        if random.randint(0, 30) == 0:
            print("loss=", (sum(meanloss) / len(meanloss)))

    print("train accuracy=", 100.0 * correct / total)
    if correct > 0.98 * total:
        break

print("keep only class 0 and 1")
X0, X1 = [], []
for x, y in trainloader:
    for i in range(x.shape[0]):
        if y[i] == 0:
            X0.append(x[i])
        if y[i] == 1:
            X1.append(x[i])
X0, X1 = torch.stack(X0, dim=0), torch.stack(X1, dim=0)

XT0, XT1 = [], []
for x, y in testloader:
    for i in range(x.shape[0]):
        if y[i] == 0:
            XT0.append(x[i])
        if y[i] == 1:
            XT1.append(x[i])
XT0, XT1 = torch.stack(XT0, dim=0), torch.stack(XT1, dim=0)

print("accuracy")
net.classifier = poisonfrog_reimpl.trainBinary(X0, X1, net)
accu0 = eval_feature.compute_accuracyRAM(
    XT0, torch.zeros(XT0.shape[0]).long(), net, flag="sum"
)
accu1 = eval_feature.compute_accuracyRAM(
    XT1, torch.ones(XT1.shape[0]).long(), net, flag="sum"
)
print(1.0 * (accu0 + accu1) / (XT0.shape[0] + XT1.shape[0]))


print("poison frog: check one shoot attack on 100 good samples from class 0")
net.cpu()
good = []
for i in range(X0.shape[0]):
    z = net(torch.unsqueeze(XT0[i], 0))[0]
    z = torch.argmax(z)
    if z == 0:
        good.append(i)
        if len(good) == 100:
            break
XT0 = XT0[good]

eval_poisonfrog(X0, X1, XT0, net, 512)

print("================ PGD FEATURE ================")
print("create feature")
net = torchvision.models.vgg13(pretrained=True)
net.features = torch.nn.Sequential(ChannelHACK(), net.features)
net.avgpool = torch.nn.Identity()
net.classifier = torch.nn.Linear(512, 100)
net = net.cuda()
net.train()

print("train setting")
import collections
import random

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.00001)
meanloss = collections.deque(maxlen=200)
nbepoch = 64

print("train")
net.cuda()
for epoch in range(nbepoch):
    print("epoch=", epoch, "/", nbepoch)
    total, correct = 0, 0
    for inputs, targets in finetuneloader:
        inputs, targets = inputs.cuda(), targets.cuda()

        pgd = eval_feature.pgd_attack(net, inputs, targets)

        outputs = net(pgd)

        loss = criterion(outputs, targets)
        meanloss.append(loss.cpu().detach().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        total += targets.shape[0]
        correct += (predicted == targets).float().sum()

        if random.randint(0, 30) == 0:
            print("loss=", (sum(meanloss) / len(meanloss)))

    print("train accuracy=", 100.0 * correct / total)
    if correct > 0.98 * total:
        break

print("keep only class 0 and 1")
X0, X1 = [], []
for x, y in trainloader:
    for i in range(x.shape[0]):
        if y[i] == 0:
            X0.append(x[i])
        if y[i] == 1:
            X1.append(x[i])
X0, X1 = torch.stack(X0, dim=0), torch.stack(X1, dim=0)

XT0, XT1 = [], []
for x, y in testloader:
    for i in range(x.shape[0]):
        if y[i] == 0:
            XT0.append(x[i])
        if y[i] == 1:
            XT1.append(x[i])
XT0, XT1 = torch.stack(XT0, dim=0), torch.stack(XT1, dim=0)

print("accuracy")
net.classifier = poisonfrog_reimpl.trainBinary(X0, X1, net)
accu0 = eval_feature.compute_accuracyRAM(
    XT0, torch.zeros(XT0.shape[0]).long(), net, flag="sum"
)
accu1 = eval_feature.compute_accuracyRAM(
    XT1, torch.ones(XT1.shape[0]).long(), net, flag="sum"
)
print(1.0 * (accu0 + accu1) / (XT0.shape[0] + XT1.shape[0]))


print("poison frog: check one shoot attack on 100 good samples from class 0")
net.cpu()
good = []
for i in range(X0.shape[0]):
    z = net(torch.unsqueeze(XT0[i], 0))[0]
    z = torch.argmax(z)
    if z == 0:
        good.append(i)
        if len(good) == 100:
            break
XT0 = XT0[good]

eval_poisonfrog(X0, X1, XT0, net, 512)
