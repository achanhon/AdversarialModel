

print("TEST.PY")
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from sklearn.metrics import confusion_matrix
import torchvision
import torchvision.transforms as transforms
device = "cuda" if torch.cuda.is_available() else "cpu"

print("load data")
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(root='./build/data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

print("load model")
net = torch.load("build/model.pth")
net = net.to(device)
net.eval()
if device == "cuda":
    torch.cuda.empty_cache()
    cudnn.benchmark = True

print("do test")
cm = np.zeros((len(classes),len(classes)),dtype=int)
with torch.no_grad():
    for _, (inputs, targets) in enumerate(testloader):
        inputs = inputs.to(device)
        outputs = net(inputs)
        _,pred = outputs.max(1)
        cm += confusion_matrix(pred.cpu().numpy(),targets.cpu().numpy(),list(range(len(classes))))

print("accuracy=",np.sum(cm.diagonal())/(np.sum(cm)+1))
print(cm)
