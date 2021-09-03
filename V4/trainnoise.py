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
from sklearn import svm
import numpy
import random


def trainClassifierOnFrozenfeatureWithNoise(
    batchprovider, encoder, datasetsize, featuredim, nbclasses, noise=50
):
    encoder.cuda()
    print("extract features")
    X = numpy.zeros((datasetsize, featuredim))
    Y = numpy.zeros(datasetsize)

    with torch.no_grad():
        i = 0
        for x, y in batchprovider:
            x, y = x.cuda(), y.cuda()
            feature = encoder(x)
            lenx = x.shape[0]
            X[i : i + lenx] = feature.cpu().numpy()
            Y[i : i + lenx] = y.cpu().numpy()
            i += lenx

    print("add label noise")
    for i in range(X.shape[0]):
        if random.randint(0, noise) == 0:
            Y[i] = random.randint(0, nbclasses)

    print("solve SVM", datasetsize, featuredim)
    classifier = svm.LinearSVC()
    classifier.fit(X, Y)

    print("extract torch classifier")
    classifierNN = torch.nn.Linear(featuredim, nbclasses)
    with torch.no_grad():
        if nbclasses > 2:
            classifierNN.weight.data = torch.Tensor(classifier.coef_)
            classifierNN.bias.data = torch.Tensor(classifier.intercept_)
        else:
            classifierNN.weight.data[0] = -torch.Tensor(classifier.coef_)
            classifierNN.bias.data[0] = -torch.Tensor(classifier.intercept_)
            classifierNN.weight.data[1] = torch.Tensor(classifier.coef_)
            classifierNN.bias.data[1] = torch.Tensor(classifier.intercept_)
    return classifierNN


print("load data")
finetuneset = torchvision.datasets.CIFAR100(
    root="./build/data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
trainset = torchvision.datasets.CIFAR10(
    root="./build/data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
testset = torchvision.datasets.CIFAR10(
    root="./build/data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
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
trainsize = eval_feature.sizeDataset("cifar", True)
testsize = eval_feature.sizeDataset("cifar", False)


print("load data")
trainset = torchvision.datasets.CIFAR10(
    root="./build/data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
testset = torchvision.datasets.CIFAR10(
    root="./build/data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
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
trainsize = eval_feature.sizeDataset("cifar", True)
testsize = eval_feature.sizeDataset("cifar", False)


print("================ NAIVE FEATURE ================")
print("create feature")
net = torchvision.models.vgg13(pretrained=True)
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

print("eval feature")
net.classifier = torch.nn.Identity().cuda()
net.classifier = trainClassifierOnFrozenfeatureWithNoise(
    trainloader, net, trainsize, 512, 10
)
print(
    "accuracy",
    eval_feature.compute_accuracy(testloader, net, testsize),
)


print("================ FSGM FEATURE ================")
print("create feature")
net = torchvision.models.vgg13(pretrained=True)
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

        fsgm = eval_feature.fsgm_attack(net, inputs, targets)

        outputs = net(fsgm)

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

print("eval feature")
net.classifier = torch.nn.Identity().cuda()
net.classifier = trainClassifierOnFrozenfeatureWithNoise(
    trainloader, net, trainsize, 512, 10
)
print(
    "accuracy",
    eval_feature.compute_accuracy(testloader, net, testsize),
)


print("================ PGD FEATURE ================")
print("create feature")
net = torchvision.models.vgg13(pretrained=True)
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

print("eval feature")
net.classifier = torch.nn.Identity().cuda()
net.classifier = trainClassifierOnFrozenfeatureWithNoise(
    trainloader, net, trainsize, 512, 10
)
print(
    "accuracy",
    eval_feature.compute_accuracy(testloader, net, testsize),
)
