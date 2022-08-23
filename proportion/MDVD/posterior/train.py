import sys
import torch
import torchvision
import density

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
Bs = 256
trainset = density.MDVD("train")
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
        net.fc = torch.nn.Linear(2048, 2)
    else:
        net.fc = torch.nn.Linear(512, 2)
else:
    if backbone == "vgg13":
        net = torchvision.models.vgg13(pretrained=True)
    else:
        net = torchvision.models.vgg16(pretrained=True)
    net.avgpool = torch.nn.Identity()
    net.classifier = torch.nn.Linear(512, 2)
net = net.cuda()
net.train()

print("train setting")
weight = torch.Tensor([1, 10]).cuda()
criterion = torch.nn.CrossEntropyLoss(weight=weight)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
nbepoch = 40

print("train")
for epoch in range(nbepoch):
    print("epoch=", epoch, "/", nbepoch)
    net.train()
    total, correct = torch.zeros(1).cuda(), torch.zeros(1).cuda()
    printloss = torch.zeros(2).cuda()
    for inputs, targets, _ in trainloader:
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
    if correct > 0.99 * total:
        break

print("reweighting")
with torch.no_grad():
    preddensity = []
    truedensity = []
    for inputs, targets, sizes in trainloader:
        inputs, targets, sizes = inputs.cuda(), targets.cuda(), sizes.cuda()
        outputs = net(inputs)

        for i in range(sizes.shape[0]):
            if targets[i] == 1:
                truedensity.append(sizes[i])
            if outputs[i][1] > outputs[i][0]:
                preddensity.append((outputs[i], sizes[i]))

    truedensity = torch.Tensor(truedensity)
    truedensity = torch.sqrt(truedensity).int()
    tmp = torch.zeros(200).cuda()
    for i in range(truedensity.shape[0]):
        tmp[truedensity[i]] = 1 + tmp[truedensity[i]]
    truedensity = normalize(smooth(tmp))

    logit = [i for i, _ in preddensity]
    logit = torch.stack(logit, dim=0)
    weight1 = torch.nn.functional.softmax(logit, dim=1)[:, 1] - 0.5
    weigth2 = torch.nn.functional.relu(logit)
    weigth2 = weigth2[:, 1] / (weigth2[:, 1] + weigth2[:, 0] + 0.01)

    preddensity = [i for _, i in preddensity]
    preddensity = torch.Tensor(preddensity)
    preddensity = torch.sqrt(preddensity).int()

    tmp = torch.ones(200).cuda() * 0.00001
    for i in range(logit.shape[0]):
        tmp[preddensity[i]] += weight1[i] + weigth2[i]
    preddensity = normalize(smooth(tmp))

    torch.save(truedensity / preddensity, "build/model_externalweigths.pth")
