print("TEST.PY")
import torch
import torch.backends.cudnn as cudnn

device = "cuda" if torch.cuda.is_available() else "cpu"

import torchvision

print("load data")
transform_test = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),
    ]
)

testset = torchvision.datasets.CIFAR10(
    root="./build/data", train=False, download=True, transform=transform_test
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

# cm = np.zeros((10, 10), dtype=int)
# with torch.no_grad():
# for _, (inputs, targets) in enumerate(testloader):
# inputs = inputs.to(device)
# outputs = net(inputs)
# _, pred = outputs.max(1)
# cm += confusion_matrix(
# pred.cpu().numpy(), targets.cpu().numpy(), list(range(10))
# )

# print("accuracy=", np.sum(cm.diagonal()) / (np.sum(cm) + 1))
# print(cm)
