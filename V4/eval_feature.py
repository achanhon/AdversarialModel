import torch


def sizeDataset(name, train):
    if name == "cifar" and train:
        return 50000
    if name == "cifar" and not train:
        return 10000

    if name == "svhn" and train:
        return 73257
    if name == "svhn" and not train:
        return 26032

    if name == "mnist" and train:
        return 60000
    if name == "mnist" and not train:
        return 10000

    print("unknown dataset")
    quit()


def compute_accuracy(batchprovider, net, datasetsize):
    net.cuda()
    with torch.no_grad():
        net = net.cuda()
        net.eval()
        accuracy = []
        for x, y in batchprovider:
            x, y = x.cuda(), y.cuda()
            _, z = (net(x)).max(1)
            accuracy.append((y == z).float().sum())
        accuracy = torch.Tensor(accuracy).sum() / datasetsize
        return accuracy.cpu().numpy()


def fsgm_attack(net, x, y, radius=3.0 / 255):
    if __debug__:
        return x
    net.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    x.requires_grad = True
    opt = torch.optim.Adam([x], lr=1)

    loss = criterion(net(x), y)
    opt.zero_grad()
    loss.backward()

    adv_x = x + radius * x.grad.sign()
    return torch.clamp(adv_x, min=0, max=1)


def pgd_attack(net, x, y, radius=3.0 / 255, alpha=0.333, iters=40):
    alpha = alpha * radius
    if __debug__:
        return x
    net.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    original_x = x

    for i in range(iters):
        x.requires_grad = True
        opt = torch.optim.Adam([x], lr=1)

        loss = criterion(net(x), y)
        opt.zero_grad()
        loss.backward()

        adv_x = x + alpha * x.grad.sign()
        eta = torch.clamp(adv_x - original_x, min=-radius, max=radius)
        x = (original_x + eta).detach_()
    return torch.clamp(x, min=0, max=1)


def compute_robust_accuracy(batchprovider, net, datasetsize, radius):
    net.cuda()
    net.eval()
    accuracy = []
    for x, y in batchprovider:
        x, y = x.cuda(), y.cuda()

        xx = pgd_attack(net, x, y, radius=radius)

        _, z = (net(xx)).max(1)
        accuracy.append((y == z).float().sum())
    accuracy = torch.Tensor(accuracy).sum() / datasetsize
    return accuracy.cpu().detach().numpy()


from sklearn import svm
import numpy


def trainClassifierOnFrozenfeature(
    batchprovider, encoder, datasetsize, featuredim, nbclasses
):
    if __debug__:
        return torch.nn.Linear(featuredim, nbclasses)
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

    print("solve SVM", datasetsize, featuredim)
    classifier = svm.LinearSVC()
    classifier.fit(X, Y)

    print("extract torch classifier")
    classifierNN = torch.nn.Linear(featuredim, nbclasses)
    with torch.no_grad():
        classifierNN.weight.data = torch.Tensor(classifier.coef_)
        classifierNN.bias.data = torch.Tensor(classifier.intercept_)
    return classifierNN


if __name__ == "__main__":
    print("example of usage")
    import torchvision
    import os

    os.system("cp -r build/data data")
    os.system("rm -r build")
    os.system("mkdir build")
    os.system("mv data build")

    print("import pretrained model")
    net = torchvision.models.vgg13(pretrained=True)
    net.avgpool = torch.nn.Identity()
    net.classifier = torch.nn.Identity()
    net.cuda()

    print("import dataset")
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
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=True, num_workers=2
    )

    print("train classifier on the top of the encoder")
    net.classifier = train_frozenfeature_classifier(
        trainloader, net, sizeDataset("cifar", True), 512, 10
    )

    print("eval")
    print(
        "train accuracy",
        compute_accuracy(trainloader, net, sizeDataset("cifar", True)),
    )
    print(
        "accuracy",
        compute_accuracy(testloader, net, sizeDataset("cifar", False)),
    )
