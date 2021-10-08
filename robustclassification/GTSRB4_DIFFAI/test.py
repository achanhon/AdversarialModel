print("TEST.PY")
import torch
import torch.backends.cudnn as cudnn

device = "cuda" if torch.cuda.is_available() else "cpu"

import torchvision
import sys

sys.path.append("..")
import computeaccuracy

print("load data")
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor(),
    ]
)
testset = torchvision.datasets.ImageFolder(
    root="/media/achanhon/bigdata/data/GTSRB/test", transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=True, num_workers=2
)

print("load model")
sys.path.append("/home/achanhon/github/diffai")
import models

net = torch.load("build/model.pth")
net = net.to(device)
net.eval()
if device == "cuda":
    torch.cuda.empty_cache()
    cudnn.benchmark = True

print("do test")
total, correct = 0, 0
with torch.no_grad():
    for inputs, targets in testloader:
        inputs, targets = inputs.to(device), targets.to(device)

        raw = net(inputs).vanillaTensorPart().detach()
        _, predicted = raw.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
print("clean_accuracy=", correct * 100 / total)
