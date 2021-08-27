import torch
import torchvision

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
classifierloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=2
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=True, num_workers=2
)

net = torchvision.models.vgg13(pretrained=True)
net.avgpool = torch.nn.Identity()
net.classifier = torch.nn.Identity()

ftrain = open("build/train.txt", "w")
with torch.no_grad():
    for x, y in classifierloader:
        x = x.to(device)
        z = net(x)
        for i in range(y.shape[0]):
            ftrain.write(str(y[i]) + " ")
            for j in range(512):
                if z[i][j] != 0.0:
                    ftrain.write(str(j + 1) + ":" + str(z[i][j]) + " ")
            ftrain.write("513:1\n")
ftrain.close()

ftest = open("build/test.txt", "w")
with torch.no_grad():
    for x, y in testloader:
        x = x.to(device)
        z = net(x)
        for i in range(y.shape[0]):
            ftrain.write(str(y[i]) + " ")
            for j in range(512):
                if z[i][j] != 0.0:
                    ftrain.write(str(j + 1) + ":" + str(z[i][j]) + " ")
            ftrain.write("513:1\n")
ftest.close()
