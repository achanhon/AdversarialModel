import torch
import torchvision

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
else:
    print("no cuda")
    quit()


print("load data")
print("load data")
raw = torchvision.transforms.ToTensor()
root, Tr, Fl, Bs = "./build/data", True, False, 128
testset = torchvision.datasets.CIFAR10(root=root, train=Fl, download=Tr, transform=raw)
testloader = torch.utils.data.DataLoader(testset, batch_size=Bs, shuffle=False)

print("load model")
net = torch.load("build/model.pth")
net = net.cuda()
net.eval()

print("test")
with torch.no_grad():
    cm = torch.zeros(10, 10).cuda()
    for inputs, targets in testloader:
        inputs, targets = inputs.cuda(), targets.cuda()

        outputs = net(inputs)
        _, predicted = outputs.max(1)

        for i in range(10):
            for j in range(10):
                cm[i][j] += torch.sum((targets == i).float() * (predicted == j).float())

    print("test cm", cm)
    total = torch.sum(cm)
    accuracy = torch.sum(torch.diagonal(cm))
    print("test accuracy=", 100.0 * accuracy / total)

    estimatedensity = cm.sum(dim=1) / total
    truedistribution = cm.sum(dim=0) / total
    print("predicted distribution", estimatedensity)
    print("true distribution", truedistribution)
    print(torch.nn.functional.kl_div(estimatedensity, truedensity))
