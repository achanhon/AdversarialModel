print("TEST.PY")
import torch
import torch.backends.cudnn as cudnn

device = "cuda" if torch.cuda.is_available() else "cpu"

import torchvision

print("load data")
testset = torchvision.datasets.CIFAR10(
    root="./build/data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=False, num_workers=2
)

print("load model")
net = torch.load("build/model.pth")
net = net.to(device)
net.eval()
if device == "cuda":
    torch.cuda.empty_cache()
    cudnn.benchmark = True

print("do test")
import sys

sys.path.append("..")
import computeaccuracy

clean_accuracy, robust_accuracy = computeaccuracy.accuracy(net, testloader)
print("clean_accuracy=", clean_accuracy, "robust_accuracy=", robust_accuracy)
