import os

os.system("cp -r build/data data")
os.system("rm -r build")
os.system("mkdir build")
os.system("mv data build")

print("================ CREATE CIFAR FEATURE ================")
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
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=2
)

print("load model")
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

    torch.save(net, "build/model.pth")
    print("train accuracy=", 100.0 * correct / total)
    if correct > 0.98 * total:
        break
    sleep(3)

del trainset, trainloader, criterion, meanloss, optimizer

print("================ EVAL CIFAR FEATURE ON MNIST ================")
import eval_feature

print("import dataset")
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.Grayscale(num_output_channels=3),
        torchvision.transforms.ToTensor(),
    ]
)
trainset = torchvision.datasets.MNIST(
    root="./build/data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=2
)

print("train classifier on MNIST the top of the CIFAR encoder")
net.classifier = torch.nn.Identity()

classifier = eval_feature.train_frozenfeature_classifier(
    trainloader, net, sizeclassicaldataset("mnist", True), 512, 10
)

print("eval classifier")
cifarnet = torch.nn.Sequential(net, classifier)
print(
    "train accuracy",
    compute_accuracy(trainloader, cifarnet, sizeclassicaldataset("mnist", True)),
)
testset = torchvision.datasets.MNIST(
    root="./build/data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=True, num_workers=2
)
print(
    "accuracy = ",
    compute_accuracy(testloader, cifarnet, sizeclassicaldataset("mnist", False)),
)

del net, classifier, cifarnet

print("================ COMPARE TO IMAGENET ONE ================")


print("train classifier on MNIST the top of the IMAGENET encoder")
net = torchvision.models.vgg13(pretrained=True)
net.avgpool = torch.nn.Identity()
net.classifier = torch.nn.Identity()
net.cuda()

classifier = eval_feature.train_frozenfeature_classifier(
    trainloader, net, sizeclassicaldataset("mnist", True), 512, 10
)

print("eval classifier")
imagenet = torch.nn.Sequential(net, classifier)
print(
    "train accuracy",
    compute_accuracy(trainloader, imagenet, sizeclassicaldataset("mnist", True)),
)
testset = torchvision.datasets.MNIST(
    root="./build/data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=True, num_workers=2
)
print(
    "accuracy = ",
    compute_accuracy(testloader, imagenet, sizeclassicaldataset("mnist", False)),
)
