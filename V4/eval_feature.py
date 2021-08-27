import torch


def sizeclassicaldataset(name, train):
    if name == "mnist" and train:
        return 60000
    if name == "mnist" and not train:
        return 10000

    if name == "cifar10" and train:
        return 50000
    if name == "cifar10" and not train:
        return 10000

    if name == "svhn" and train:
        return 73257
    if name == "svhn" and not train:
        return 26032

    print("unknown dataset")
    quit()


def compute_accuracy(batchprovider, net, datasetsize, device="cuda"):
    with torch.no_grad():
        net.to(device)
        net.eval()
        accuracy = []
        for x, y in batchprovider:
            x, y = x.to(device), y.to(device)
            _, z = (net(x)).max(1)
            accuracy.append((y == z).float().sum())
        accuracy = torch.Tensor(accuracy).sum() / datasetsize
        return accuracy.cpu().detach().numpy()


from sklearn import svm
import numpy


def train_frozenfeature_classifier(
    batchprovider, encoder, datasetsize, featuredim, nbclasses, device="cuda"
):
    print("extract features")
    X = numpy.zeros((datasetsize, featuredim))
    Y = numpy.zeros(datasetsize)

    with torch.no_grad():
        i = 0
        for x, y in batchprovider:
            x, y = x.to(device), y.to(device)
            feature = encoder(x)
            lenx = x.shape[0]
            X[i : i + lenx] = feature.cpu().numpy()
            Y[i : i + lenx] = y.cpu().numpy()
            i += lenx

    print("solve SVM", datasetsize, featuredim)
    classifier = svm.LinearSVC()
    classifier.fit(X * 100, Y)

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

    print("import dataset")
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor(),
        ]
    )
    trainset = torchvision.datasets.SVHN(
        root="./build/data", split="train", download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=2
    )

    print("import pretrained model")
    encoder = torchvision.models.vgg13(pretrained=True)
    encoder.avgpool = torch.nn.Identity()
    encoder.classifier = torch.nn.Identity()
    encoder.cuda()

    print("train classifier on the top of the encoder")
    classifier = train_frozenfeature_classifier(
        trainloader, encoder, sizeclassicaldataset("svhn", True), 512, 10
    )

    print("eval classifier")
    net = torch.nn.Sequential(encoder, classifier)
    print(
        "train accuracy",
        compute_accuracy(trainloader, net, sizeclassicaldataset("svhn", True)),
    )
    testset = torchvision.datasets.SVHN(
        root="./build/data", split="test", download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=True, num_workers=2
    )
    print(
        "accuracy = ",
        compute_accuracy(testloader, net, sizeclassicaldataset("svhn", False)),
    )
