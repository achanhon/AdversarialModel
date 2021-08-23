import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

print("TRAIN DATA")
import torchvision
import torchvision.transforms as transforms

transform_train = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)
trainset = torchvision.datasets.CIFAR10(
    root="./build/data", train=True, download=True, transform=transform_train
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=1, shuffle=False, num_workers=2
)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

print("MODEL")
print("all models are built from")
net = torchvision.models.vgg19(pretrained=False, progress=True)
del net
print("here a bag of models is used to poison data")
hackmodel1 = torch.load("build/hackmodel1.pth")
hackmodel2 = torch.load("build/hackmodel2.pth")
hackmodel3 = torch.load("build/hackmodel3.pth")
hackmodel4 = torch.load("build/hackmodel4.pth")
hackmodel5 = torch.load("build/hackmodel5.pth")
hackmodels = (hackmodel1, hackmodel2, hackmodel3, hackmodel4, hackmodel5)
for j in range(5):
    hackmodels[j].to(device)
    hackmodels[j].eval()

fairmodel1 = torch.load("build/fairmodel1.pth")
fairmodel2 = torch.load("build/fairmodel2.pth")
fairmodel3 = torch.load("build/fairmodel3.pth")
fairmodel4 = torch.load("build/fairmodel4.pth")
fairmodel5 = torch.load("build/fairmodel5.pth")
fairmodels = (fairmodel1, fairmodel2, fairmodel3, fairmodel4, fairmodel5)
for j in range(5):
    fairmodels[j].to(device)
    fairmodels[j].eval()

print("DEFINE POISONING")
print("forward-backward data to update DATA not the weight: this is poisonning !")
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import PIL
import PIL.Image

criterion = nn.CrossEntropyLoss()
averagepixeldiff = []
manormalisation = transforms.Normalize(
    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), False
)

for batch_idx, (inputs, targets) in enumerate(trainloader):
    x = inputs.clone()
    x.requires_grad = True

    # simulate crops
    x4 = F.pad(x, (4, 4, 4, 4))
    batch = []
    for i in range(7):
        row = random.randint(0, 3)
        col = random.randint(0, 3)
        xx = x4[:, :, row : row + 32, col : col + 32]
        if random.randint(0, 2) == 0:
            xx = torch.flip(xx, [3])

        xx = manormalisation(xx[0])
        batch.append(xx)
    batch = torch.stack(batch, dim=0).to(device)
    targets = torch.cat([targets] * 7, dim=0).to(device)

    # goal is to modify batch such that batch is "ok" for hackedmodel but not for fair model
    hackgradient = np.zeros((5, 3, 32, 32), dtype=int)
    fairgradient = np.zeros((5, 3, 32, 32), dtype=int)

    for j in range(5):
        tmpbatch = batch.clone()
        optimizer = optim.SGD([x], lr=1, momentum=0)
        optimizer.zero_grad()

        outputs = hackmodels[j](tmpbatch)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)

        hackgradient[j] = np.sign(x.grad.cpu().numpy())[0]

    for j in range(5):
        tmpbatch = batch.clone()
        optimizer = optim.SGD([x], lr=1, momentum=0)
        optimizer.zero_grad()

        outputs = fairmodels[j](tmpbatch)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)

        fairgradient[j] = np.sign(x.grad.cpu().numpy())[0]

    xgrad = (
        fairgradient[0]
        + fairgradient[1]
        + fairgradient[2]
        + fairgradient[3]
        + fairgradient[4]
    ) - (
        hackgradient[0]
        + hackgradient[1]
        + hackgradient[2]
        + hackgradient[3]
        + hackgradient[4]
    )
    xgrad = np.sign(xgrad)

    xpoison = inputs.cpu().numpy()[0].copy() * 255 - xgrad
    xpoison = np.minimum(xpoison, np.ones(xpoison.shape, dtype=float) * 255)
    xpoison = np.maximum(xpoison, np.zeros(xpoison.shape, dtype=float))

    averagepixeldiff.append(np.sum(np.abs(xgrad)))
    im = np.transpose(xpoison, (1, 2, 0))
    im = PIL.Image.fromarray(np.uint8(im))
    im.save("build/poison/" + str(int(targets[0])) + "/" + str(batch_idx) + ".png")

    if batch_idx % 500 == 499:
        print(
            batch_idx,
            "/",
            len(trainloader),
            " mean diff",
            sum(averagepixeldiff) / len(averagepixeldiff) / 3 / 32 / 32,
        )
