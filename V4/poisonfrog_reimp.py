import torch
import eval_feature


def find_candidate_for_collision(X, Y, encoder, xt, yt, radius):
    with torch.no_grad():
        zt = encoder(xt)

    bestgap, candidate, candidateafterattack = None, None, None
    for i in range(X.shape[0]):
        if Y[i] == yt:
            continue
        x = X[i].clone().view(xt.shape)

        for j in range(3):
            x.requires_grad = True
            opt = torch.optim.SGD([x], lr=1)
            z = net(x)

            gap = torch.sum((z - zt).abs())
            opt.zero_grad()
            gap.backward()

            adv_x = x - radius / 3 * x.grad.sign()
            adv_x = torch.clamp(adv_x, min=0, max=1)

        eta = torch.clamp(adv_x - original_x, min=-radius, max=radius)
        adv_x = (x + eta).detach_()
        with torch.no_grad():
            z = net(adv_x)
            gap = torch.sum((z - zt).abs())

        if bestgap is None or gap < bestgap:
            candidate, candidateafterattack = i, adv_x

    return candidate, candidateafterattack


def eval_poisonfrog(X, Y, Xtest, Ytest, net, featuredim, nbclasses, radius=3.0 / 255):
    successful_attack = 0
    for i in range(Xtest.shape[0]):
        print(i, "/100")
        xt, yt = Xtest[i].clone(), Ytest[i]
        xt = xt.view(1, xt.shape[0], xt.shape[1], xt.shape[2])

        net.classifier = torch.nn.Identity()
        net.cpu()
        candidate, candidateafterattack = find_candidate_for_collision(
            X, Y, net, xt, yt, radius
        )

        print("learn with X being modified")
        xBACKUP = X[candidate].clone()
        X[candidate] = candidateafterattack
        poisonnedDataset = torch.utils.data.TensorDataset(X, Y)
        poisonnedloader = torch.utils.data.DataLoader(
            poisonnedDataset, batch_size=64, shuffle=True, num_workers=2
        )

        net.classifier = torch.nn.Identity()
        net.cuda()
        net.classifier = eval_feature.trainClassifierOnFrozenfeature(
            poisonnedloader, net, X.shape[0], featuredim, nbclasses
        )

        net.cpu()
        zt = net(xt)[0]
        zt = torch.argmax(zt)
        if zt != yt:
            successful_attack += 1
            print("good shot")
        else:
            print(":-(")

        X[candidate] = xBACKUP

    print(successful_attack)


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
        train=False,  ### hack
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    trainsize = eval_feature.sizeDataset("cifar", False)
    testset = torchvision.datasets.CIFAR10(
        root="./build/data",
        train=True,  ### hack
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    testsize = eval_feature.sizeDataset("cifar", False)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=True, num_workers=2
    )

    print("train classifier on the top of the encoder")
    net.classifier = eval_feature.trainClassifierOnFrozenfeature(
        trainloader, net, trainsize, 512, 10
    )
    print(
        "accuracy",
        eval_feature.compute_accuracy(testloader, net, testsize),
    )

    print("collect 10 targets per classes")
    Xt = [[] for i in range(10)]
    for x, y in testloader:
        for i in range(x.shape[0]):
            Xt[y[i]].append(x[i].view(1, x.shape[1], x.shape[2], x.shape[3]))

    net.cpu()
    with torch.no_grad():
        for i in range(10):
            tmp = []
            for x in Xt[i]:
                z = net(x)[0]
                z = torch.argmax(z)
                if z == i:
                    tmp.append(x)
                    if len(tmp) == 10:
                        Xt[i] = tmp
                        break

    tmp = []
    for i in range(10):
        tmp += Xt[i]
    Xt = torch.cat(tmp, dim=0)
    _, Yt = torch.max(net(Xt), dim=1)
    print(Xt.shape[0])

    print("change dataset shape")
    Y = torch.zeros(trainsize)
    X = torch.zeros((trainsize, 3, 32, 32))
    i = 0
    for x, y in trainloader:
        lenx = x.shape[0]
        X[i : i + lenx] = x
        Y[i : i + lenx] = y
        i += lenx

    print("poison frog")
    X, Y, Xt, Yt = X.cpu(), Y.cpu(), Xt.cpu(), Yt.cpu()
    eval_poisonfrog(X, Y, Xt, Yt, net, 512, 10)
