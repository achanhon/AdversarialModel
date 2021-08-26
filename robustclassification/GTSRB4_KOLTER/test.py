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

if computeaccuracy.isonspiro():
    testset = torchvision.datasets.ImageFolder(
        root="/scratchm/achanhon/GTSRB_miseenforme/test", transform=transform
    )
else:
    testset = torchvision.datasets.ImageFolder(
        root="/data/GTSRB_misenforme/test", transform=transform
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
clean_accuracy, robust_accuracy = computeaccuracy.accuracy(net, testloader)
print("clean_accuracy=", clean_accuracy, "robust_accuracy=", robust_accuracy)
