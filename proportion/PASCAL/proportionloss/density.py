import os
import PIL
from PIL import Image
import torch
import torchvision


def smooth(vect):
    return torch.nn.functional.avg_pool1d(
        vect.unsqueeze(0).unsqueeze(0), kernel_size=31, padding=15, stride=1
    )[0][0]


def normalize(positive_vect):
    total = positive_vect.sum()
    return positive_vect / total


def logitTOdensity(logit, sizes):
    density = torch.ones(450) * 0.0001

    weight1 = torch.nn.functional.softmax(logit, dim=1)[:, 1] - 0.5
    weigth2 = torch.nn.functional.relu(logit)
    weigth2 = weigth2[:, 1] / (weigth2[:, 1] + weigth2[:, 0] + 0.01)
    weigth2 = weigth2 + weight1

    I = [i for i in range(sizes.shape[0]) if weight1[i] > 0]
    for i in I:
        density[sizes[i]] = weigth2[i] + density[sizes[i]]

    return normalize(smooth(density))


def labelsT0density(targets, sizes):
    density = torch.ones(450) * 0.0001

    I = [i for i in range(sizes.shape[0]) if targets[i] == 1]
    for i in I:
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
        if size >= 450 * 450:
            size = 450 * 450 - 1
        return self.aug(image), label, size
