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
    print("no output provided")
    quit()

if len(sys.argv) > 2:
    print("just merge result")
    outputpath = sys.argv[1]
    inputpaths = sys.argv[2:]
    inputs = [torch.load("build/" + name) for name in inputpaths]
    print(inputs)
    inputs = torch.Tensor(inputs)
    mean = inputs.mean()
    var = inputs.var()
    meanvar = str(mean.cpu().numpy()) + "  (\pm " + str(var.cpu().numpy()) + ")\n"
    with open(outputpath, "w") as f:
        f.write(meanvar)
    f.close()
    print(mean, var)
    quit()


print("load data")
Bs = 256
testset = density.MDVD("test")
testloader = torch.utils.data.DataLoader(testset, batch_size=Bs, shuffle=True)

print("load model")
net = torch.load("build/model.pth")
weights = torch.load("build/model_externalweigths.pth")
net = net.cuda()
net.eval()

with torch.no_grad():
    print("classical test")
    cm = torch.zeros(2, 2).cuda()
    for inputs, targets, _ in testloader:
        inputs, targets = inputs.cuda(), targets.cuda()

        outputs = net(inputs)
        _, predicted = outputs.max(1)

        for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            cm[i][j] += torch.sum((targets == i).float() * (predicted == j).float())

    total = torch.sum(cm)
    accuracy = torch.sum(torch.diagonal(cm)) / total
    print("test cm", cm)
    print("test accuracy=", accuracy)

    print("proportion test")
    total, averageKL, KLsele, KLfair = 0, 0, 0, 0
    for epoch in range(10):
        for inputs, targets, sizes in testloader:
            sizes = torch.sqrt(sizes).int()
            truedensity = density.labelsT0density(targets, sizes)

            inputs = inputs.cuda()
            outputs = net(inputs).cpu()

            estimatedensity = density.logitTOdensity(outputs, sizes)
            withrejection = density.selectivelogitTOdensity(outputs, sizes)
            withfairness = normalize(estimatedensity.clone() * weights)

            averageKL = density.extendedKL(estimatedensity, truedensity) + averageKL
            KLsele = density.extendedKL(withrejection, truedensity) + KLsele
            KLfair = density.extendedKL(withfairness, truedensity) + KLfair
            total += 1

    averageKL = averageKL / total
    KLsele = KLsele / total
    KLfair = KLfair / total
    torch.save(averageKL, "build/baseline_" + sys.argv[1])
    torch.save(KLsele, "build/selective_" + sys.argv[1])
    torch.save(KLfair, "build/fair_" + sys.argv[1])
    print("test divergence=", averageKL, KLsele, KLfair)
