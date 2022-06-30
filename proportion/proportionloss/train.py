import torch
import torchvision

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
else:
    print("no cuda")
    quit()

print("load data")
raw = torchvision.transforms.ToTensor()
root, Tr, Bs = "./build/data", True, 256
trainset = torchvision.datasets.CIFAR10(root=root, train=Tr, download=Tr, transform=raw)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=Bs, shuffle=True)

print("load model")
RESNET = True
if RESNET:
    net = torchvision.models.resnet50(pretrained=True)
    net.avgpool = torch.nn.Identity()
    net.fc = torch.nn.Linear(2048, 10)
else:
    net = torchvision.models.vgg16(pretrained=True)
    net.avgpool = torch.nn.Identity()
    net.classifier = torch.nn.Linear(512, 10)
net = net.cuda()
net.train()

print("train setting")
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
nbepoch = 20

print("train")
for epoch in range(nbepoch):
    print("epoch=", epoch, "/", nbepoch)
    net.train()
    total, correct = torch.zeros(1).cuda(), torch.zeros(1).cuda()
    printloss = torch.zeros(2).cuda()
    for inputs, targets in trainloader:
        inputs, targets = inputs.cuda(), targets.cuda()

        outputs = net(inputs)
        primaryloss = criterion(outputs, targets)

        estimatedensity = torch.nn.functional.softmax(outputs, dim=1)
        estimatedensity = torch.sum(estimatedensity, dim=0) / Bs

        truedensity = torch.zeros(10).cuda()
        for i in range(10):
            truedensity[i] = (targets == i).float().sum() / Bs

        if torch.abs(estimatedensity.sum() - 1) < 0.0001:
            print(estimatedensity.sum())
        assert torch.abs(estimatedensity.sum() - 1) < 0.0001
        assert torch.abs(truedensity.sum() - 1) < 0.0001

        secondaryloss = torch.nn.functional.kl_div(estimatedensity, truedensity)
        secondaryloss = torch.nn.functional.relu(secondaryloss + 0.01)

        loss = primaryloss + secondaryloss
        printloss[0] += loss.detach()
        printloss[1] += Bs

        if epoch > 5:
            loss = 0.1 * primaryloss + 10 * secondaryloss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        optimizer.step()

        _, predicted = outputs.max(1)
        total += Bs
        correct += (predicted == targets).float().sum()

        if printloss[1] > 2000:
            print("loss=", printloss[0] / printloss[1])
            printloss = torch.zeros(2).cuda()

    torch.save(net, "build/model.pth")
    print("train accuracy=", 100.0 * correct / total)
    if correct > 0.98 * total:
        quit()
