import os
import PIL
from PIL import Image
import numpy


l = os.listdir(
    "/home/achanhon/github/AdversarialModel/proportion/PASCAL/selectivesearch/build/PASCAL/train/good"
)
sizes = []
for name in l:
    image = (
        PIL.Image.open(
            "/home/achanhon/github/AdversarialModel/proportion/PASCAL/selectivesearch/build/PASCAL/train/good/"
            + name
        )
        .convert("RGB")
        .copy()
    )
    size = image.size[0] * image.size[0] + image.size[1] * image.size[1]
    sizes.append(size)

sizes = numpy.asarray(sizes)
sizes = numpy.sqrt(sizes).int()
print(numpy.mean(sizes), numpy.var(sizes), numpy.max(sizes))
quit()


import torch
import torchvision


def smooth(vect):
    return torch.nn.functional.avg_pool1d(
        vect.unsqueeze(0), kernel_size=15, padding=7, stride=1
    )[0]


def normalize(positive_vect):
    total = positive_vect.sum()
    return positive_vect / total


def logitTOdensity(logit, sizes):
    sizes = torch.sqrt(sizes).int()
    density = torch.zeros(200).cuda()

    weight1 = torch.nn.functional.softmax(logit, dim=1)[:, 1] - 0.5
    weigth2 = torch.nn.functional.relu(logit)
    weigth2 = weigth2[:, 1] / (weigth2[:, 1] + weigth2[:, 0] + 0.01)

    for i in range(sizes.shape[0]):
        if weight1[i] > 0:
            density[sizes[i]] = weight1[i] + weigth2[i] + density[sizes[i]]

    return normalize(smooth(density))


def selectivelogitTOdensity(logit, sizes):
    sizes = torch.sqrt(sizes).int()
    density = torch.zeros(200).cuda()

    weight1 = torch.nn.functional.softmax(logit, dim=1)[:, 1] - 0.5
    weigth2 = torch.nn.functional.relu(logit)
    weigth2 = weigth2[:, 1] / (weigth2[:, 1] + weigth2[:, 0] + 0.01)

    kept = []
    for i in range(weight1.shape[0]):
        if weight1[i] > 0.5:
            kept.append((weight1[i] - 0.5, i))
    kept = sorted(kept)
    kept = kept[int(0.2 * len(kept)) :]

    for i in kept:
        density[sizes[i]] = weight1[i] + weigth2[i] + density[sizes[i]]

    return normalize(smooth(density))


def labelsT0density(targets, sizes):
    sizes = torch.sqrt(sizes).int()
    density = torch.zeros(200).cuda()

    for i in range(sizes.shape[0]):
        if targets[i] == 1:
            density[sizes[i]] = 1 + density[sizes[i]]

    return normalize(smooth(density))


def extendedKL(estimatedensity, truedensity):
    kl_loss = torch.nn.KLDivLoss(reduction="sum")
    kl = kl_loss(estimatedensity, truedensity)
    kl = torch.nn.functional.relu(kl)
    diff = estimatedensity - truedensity
    return kl + torch.sum(diff * diff) + diff.abs().sum()


class PASCAL(torch.utils.data.Dataset):
    def __init__(self, flag):
        assert flag in ["train", "test"]
        self.flag = flag
        self.root = "../selectivesearch/build/PASCAL/" + flag
        tmp = [
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor(),
        ]
        self.aug = torchvision.transforms.Compose(tmp)

        self.sizeP = len(os.listdir(self.root + "/good"))
        self.sizeN = len(os.listdir(self.root + "/bad"))

    def __len__(self):
        return self.sizeP + self.sizeN

    def __getitem__(self, idx):
        if idx < self.sizeN:
            path = self.root + "/bad/" + str(idx) + ".png"
            label = 0
        else:
            path = self.root + "/good/" + str(idx - self.sizeN) + ".png"
            label = 1
        image = PIL.Image.open(path).convert("RGB").copy()

        size = image.size[0] * image.size[0] + image.size[1] * image.size[1]
        if size >= 200 * 200:
            size = 200 * 200 - 1
        return self.aug(image), label, size
