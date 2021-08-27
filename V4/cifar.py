import os

os.system("cp -r build/data data")
os.system("rm -r build")
os.system("mkdir build")
os.system("mv data build")

import torch
import torch.backends.cudnn as cudnn
import torchvision

if not torch.cuda.is_available():
    quit()
device = "cuda"
torch.cuda.empty_cache()
cudnn.benchmark = True

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
finetunesize = int(eval_feature.sizeclassicaldataset("cifar10", True) * 0.33)
classifiersize = eval_feature.sizeclassicaldataset("cifar10", True) - finetunesize
finetuneset, classifierset = torch.utils.data.random_split(
    trainset, [finetunesize, classifiersize]
)

finetuneloader = torch.utils.data.DataLoader(
    finetuneset, batch_size=64, shuffle=True, num_workers=2
)
classifierloader = torch.utils.data.DataLoader(
    classifierset, batch_size=64, shuffle=True, num_workers=2
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=True, num_workers=2
)
testsize = eval_feature.sizeclassicaldataset("cifar10", False)

print("================ CREATE CIFAR FEATURE ================")
net = torchvision.models.vgg13(pretrained=True)
net.avgpool = torch.nn.Identity()
net.classifier = torch.nn.Linear(512, 10)
net = net.to(device)
net.train()

print("train setting")
import collections
import random

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
meanloss = collections.deque(maxlen=200)
nbepoch = 8

print("train")
for epoch in range(nbepoch):
    print("epoch=", epoch, "/", nbepoch)
    total, correct = 0, 0
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
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

del finetuneset, trainloader, criterion, meanloss, optimizer

print(
    "accuracy = ",
    eval_feature.compute_accuracy(testloader, net, testsize),
)


print("================ CREATE CIFAR CLASSIFIER ================")
import eval_feature

net.classifier = torch.nn.Identity()
net.classifier = eval_feature.train_frozenfeature_classifier(
    classifierloader, net, classifiersize, 512, 10
)

print(
    "accuracy = ",
    eval_feature.compute_accuracy(testloader, net, testsize),
)

print("================ COMPARE TO IMAGENET FEATURE ================")

net = torchvision.models.vgg13(pretrained=True)
net.avgpool = torch.nn.Identity()
net.classifier = torch.nn.Identity()
net.cuda()

net.classifier = eval_feature.train_frozenfeature_classifier(
    classifierloader, net, classifiersize, 512, 10
)

print(
    "accuracy = ",
    eval_feature.compute_accuracy(testloader, net, testsize),
)
