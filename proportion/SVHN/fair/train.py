import sys
import torch
import torchvision

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
else:
    print("no cuda")
    quit()
if len(sys.argv) == 1:
    print("no backbone provided")
    quit()


print("load data")
aug = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomResizedCrop(32),
        torchvision.transforms.RandomRotation(10),
        torchvision.transforms.RandomHorizontalFlip(0.5),
        torchvision.transforms.ToTensor(),
    ]
)

root, Tr, Bs, Sp = "./build/data", True, 256, "train"
trainset = torchvision.datasets.SVHN(root=root, split=Sp, download=Tr, transform=aug)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=Bs, shuffle=True)

print("load model")
backbone = sys.argv[1]
assert backbone in ["resnet50", "resnet34", "vgg13", "vgg16"]
if "resnet" in backbone:
    if backbone == "resnet50":
        net = torchvision.models.resnet50(pretrained=True)
    else:
        net = torchvision.models.resnet34(pretrained=True)
    net.avgpool = torch.nn.Identity()
    if backbone == "resnet50":
        net.fc = torch.nn.Linear(2048, 10)
    else:
        net.fc = torch.nn.Linear(512, 10)
else:
    if backbone == "vgg13":
        net = torchvision.models.vgg13(pretrained=True)
    else:
        net = torchvision.models.vgg16(pretrained=True)
    net.avgpool = torch.nn.Identity()
    net.classifier = torch.nn.Linear(512, 10)
net = net.cuda()
net.train()

print("train setting")
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
nbepoch = 40

print("train")
for epoch in range(nbepoch):
    print("epoch=", epoch, "/", nbepoch)
    net.train()
    total, correct = torch.zeros(1).cuda(), torch.zeros(1).cuda()
    printloss = torch.zeros(2).cuda()
    for inputs, targets in trainloader:
        inputs, targets = inputs.cuda(), targets.cuda()

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        printloss[0] += loss.detach()
        printloss[1] += Bs

        if epoch > nbepoch // 2:
            loss *= 0.1

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
        break


print("reweighting")
with torch.no_grad():
    classweigth = torch.zeros(10).cuda()
    totalsize = 0
    for inputs, _ in trainloader:
        inputs = inputs.cuda()
        outputs = net(inputs)

        softmaxdensity = torch.nn.functional.softmax(outputs, dim=1)
        tmp = torch.nn.functional.relu(outputs) + softmaxdensity
        total = tmp.sum(dim=1)
        total = torch.stack([total] * 10, dim=1)
        estimatedensity = tmp / total

        classweigth = classweigth + estimatedensity.sum(dim=0)
        totalsize = totalsize + inputs.shape[0]

    classweigth = classweigth / totalsize
    print("classweigth", classweigth)
    torch.save(classweigth, "build/model_externalweigths.pth")
