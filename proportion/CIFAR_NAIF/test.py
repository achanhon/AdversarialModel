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
if len(argv) == 1:
    print("no backbone provided")
    quit()

if len(argv > 2):
    print("just merge result")
    outputpath = argv[1]
    inputpaths = argv[2:]
    inputs = [torch.load(name) for name in inputpaths]
    inputs = torch.cat(inputs)
    mean = inputs.mean()
    var = inputs.var()
    meanvar = str(mean.cpu().numpy()) + "  (\pm " + str(var.cpu().numpy()) + ")"
    with open(outputpath, "w") as f:
        f.write(meanvar)
    f.close()
    print(mean, var)
    quit()


print("load data")
raw = torchvision.transforms.ToTensor()
root, Tr, Fl, Bs = "./build/data", True, False, 256
testset = torchvision.datasets.CIFAR10(root=root, train=Fl, download=Tr, transform=raw)
testloader = torch.utils.data.DataLoader(testset, batch_size=Bs, shuffle=True)

print("load model")
net = torch.load("build/model.pth")
net = net.cuda()
net.eval()

with torch.no_grad():
    print("classical test")
    cm = torch.zeros(10, 10).cuda()
    for inputs, targets in testloader:
        inputs, targets = inputs.cuda(), targets.cuda()

        outputs = net(inputs)
        _, predicted = outputs.max(1)

        for i in range(10):
            for j in range(10):
                cm[i][j] += torch.sum((targets == i).float() * (predicted == j).float())

    total = torch.sum(cm)
    accuracy = torch.sum(torch.diagonal(cm)) / total
    print("test cm", cm)
    print("test accuracy=", accuracy)

    print("proportion test")
    averageKL = torch.zeros(2).cuda()
    for epoch in range(10):
        for inputs, targets in testloader:
            inputs, targets = inputs.cuda(), targets.cuda()

            outputs = net(inputs)

            estimatedensity = density.logitTOdensity(outputs)
            truedensity = density.labelsTOdensity(targets)
            averageKL[0] += density.extendedKL(estimatedensity, truedensity)
            averageKL[1] += 1

    averageKL = averageKL[0] / averageKL[1]
    torch.save(sys.argv[1], averageKL)
    print("test divergence=", averageKL)
