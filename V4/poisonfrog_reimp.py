import torch
import eval_feature


def trainBinary(X0, X1, net):
    X = torch.cat([X0, X1], dim=0).float()
    Y = torch.ones(X.shape[0])
    Y[0 : X0.shape[0]] = 0
    Y = Y.long()

    trainset = torch.torch.utils.data.TensorDataset(X, Y)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=2
    )

    classifier = eval_feature.trainClassifierOnFrozenfeature(
        trainloader, net, X.shape[0], 512, 2
    )
    return classifier


def find_candidate_for_collision(X, encoder, zt, radius):
    bestgap, candidate, candidateafterattack = None, None, None
    for i in range(X.shape[0]):
        x = torch.unsqueeze(X[i].clone(), 0)
        ori_x = x.clone()

        for j in range(5):
            x.requires_grad = True
            opt = torch.optim.SGD([x], lr=1)
            z = net(x)

            gap = torch.sum((z - zt).abs())
            opt.zero_grad()
            gap.backward()

            adv_x = x - radius / 3 * x.grad.sign()
            adv_x = torch.clamp(adv_x, min=0, max=1)

            eta = torch.clamp(adv_x - ori_x, min=-radius, max=radius)
            x = (ori_x + eta).detach_()

        with torch.no_grad():
            z = net(x)
            gap = torch.sum((z - zt).abs())

        if bestgap is None or gap < bestgap:
            bestgap = gap
            candidate, candidateafterattack = i, x

    print(bestgap)
    return candidate, candidateafterattack


def eval_poisonfrog(X0, X1, XT0, net, featuredim, radius=6.0 / 255):
    net.classifier = torch.nn.Identity()
    with torch.no_grad():
        ZT0 = net(XT0)

    successful_attack = 0
    for i in range(100):
        print(i, "/100")

        net.classifier = torch.nn.Identity()
        net.cpu()
        candidate, candidateafterattack = find_candidate_for_collision(
            X1, net, ZT0[i], radius
        )

        print("learn with one sample being modified")
        net.cuda()
        xBACKUP = X1[candidate].clone()
        X1[candidate] = candidateafterattack
        net.classifier = trainBinary(X0, X1, net)

        print("good shot ?")
        net.cpu()
        zt = net(torch.unsqueeze(XT0[i], 0))[0]
        zt = torch.argmax(zt)
        if zt == 1:
            successful_attack += 1
            print("good shot")
        else:
            print(":-(")

        X1[candidate] = xBACKUP

    print(successful_attack)


if __name__ == "__main__":
    print("example of usage")
    import torchvision
    import os

    os.system("cp -r build/data data")
    os.system("rm -r build")
    os.system("mkdir build")
    os.system("mv data build")

    print("import dataset")
    trainset = torchvision.datasets.CIFAR10(
        root="./build/data",
        train=False,  ### hack
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    testset = torchvision.datasets.CIFAR10(
        root="./build/data",
        train=True,  ### hack
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=True, num_workers=2
    )

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

    print("import pretrained model")
    net = torchvision.models.vgg13(pretrained=True)
    net.avgpool = torch.nn.Identity()
    net.classifier = torch.nn.Identity()
    net.cuda()

    print("train classifier on the top of the encoder")
    net.classifier = trainBinary(X0, X1, net)

    print("accuracy")
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
