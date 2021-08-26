import torch


def compute_accuracy(batchprovider, encoder, classifier, datasetsize, device="cuda"):
    with torch.no_grad():
        accuracy = []
        for x, y in batchprovider:
            x, y = x.to(device), y.to(device)
            feature = encoder(x)
            z = classifier(feature)
            z, _ = torch.max(z, dim=1)
            accuracy.append((y == z).float()).sum()
        accuracy = torch.Tensor(accuracy).sum() / datasetsize
        return accuracy.cpu().detach().numpy()


def train_frozenfeature_classifier(
    batchprovider,
    encoder,
    datasetsize,
    featuredim,
    nbclasses,
    nbepoch=100,
    lr=0.001,
    device="cuda",
    earlystopping=True,
):
    classifier = torch.nn.Linear(featuredim, nbclasses)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.6)

    for epoch in range(nbepoch):
        for x, y in batchprovider:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                feature = encoder(x)
            z = classifier(feature)

            loss = criterion(z, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        accuracy = compute_accuracy(
            batchprovider, encoder, classifier, datasetsize, device=device
        )
        print(epoch, accuracy)
        if earlystopping and accuracy > 0.98:
            return classifier
    return classifier


if __name__ == __main__:
    print("example of usage")
    import torchvision
    import os

    os.system("cp -r build/data data")
    os.system("rm -r build")
    os.system("mkdir build")
    os.system("mv data build")

    print("import dataset")
    trainset = torchvision.datasets.MNIST(
        root="./build/data",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2
    )

    print("import pretrained model")
    net = torchvision.models.vgg13(pretrained=True)
    net.avgpool = torch.nn.Identity()
    net.classifier = torch.nn.Identity()

    print("train classifier on the top of the encoder")

    train_frozenfeature_classifier(trainloader, net, 40000, 512, 10)

    print("eval classifier")
    testset = torchvision.datasets.MNIST(
        root="./build/data",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    testloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2
    )

    print("accuracy = ", compute_accuracy(testloader, net, classifier, 10000))
