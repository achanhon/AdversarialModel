import os
import torch
import torchvision


def normalize(positive_vect):
    total = positive_vect.sum()
    return positive_vect / total


def logitTOdensity(logit, sizes):
    sizes = torch.sqrt(sizes)
    density = torch.zeros(200).cuda()

    weight1 = torch.nn.functional.softmax(logit, dim=1) - 0.5
    weigth2 = torch.nn.functional.relu(logit)
    weigth2 = weigth2[:, 1] / (weigth2[:, 1] + weigth2[:, 0] + 0.01)

    for i in range(sizes.shape[0]):
        if weight1[i] > 0:
            density[sizes[i]] = weight1[i] + weigth2[i] + density[sizes[i]]

    density = torch.nn.functional.avg_pool1d(
        density, kernel_size=7, padding=3, stride=1
    )
    return normalize(density)


def labelsT0density(targets, sizes):
    sizes = torch.sqrt(sizes)
    density = torch.zeros(200).cuda()

    for i in range(sizes.shape[0]):
        if targets[i] == 1:
            density[sizes[i]] = 1 + density[sizes[i]]

    density = torch.nn.functional.avg_pool1d(
        density, kernel_size=7, padding=3, stride=1
    )
    return normalize(density)


def extendedKL(estimatedensity, truedensity):
    kl_loss = torch.nn.KLDivLoss(reduction="sum")
    kl = kl_loss(estimatedensity, truedensity)
    kl = torch.nn.functional.relu(kl)
    diff = estimatedensity - truedensity
    return kl + torch.sum(diff * diff) + diff.abs().sum()


class TwoHead(torch.nn.Module):
    def __init__(self, inputsize, outputsize):
        super(TwoHead, self).__init__()
        self.fc1 = torch.nn.Linear(inputsize, outputsize)
        self.fc2 = torch.nn.Linear(inputsize * 2, outputsize)

    def forward(self, x):
        xm, _ = x.max(dim=0)
        xa = x.mean(dim=0)
        xma = torch.cat([xm, xa])
        predensity = self.fc2(xma)

        softmaxdensity = torch.nn.functional.softmax(predensity)
        tmp = torch.nn.functional.relu(predensity) + softmaxdensity
        total = tmp.sum()
        estimatedensity = tmp / total

        return self.fc1(x), estimatedensity


class MDVD(torch.utils.data.Dataset):
    def __init__(self, flag):
        assert flag in ["train", "test"]
        self.flag = flag
        self.root = "../selectivesearch/build/MDVD/" + flag
        if flag == "test":
            tmp = [torchvision.transforms.Resize(32), torchvision.transforms.ToTensor()]
        else:
            tmp = [
                torchvision.transforms.RandomResizedCrop(32),
                torchvision.transforms.RandomRotation(90),
                torchvision.transforms.RandomHorizontalFlip(0.5),
                torchvision.transforms.RandomVerticalFlip(0.5),
                torchvision.transforms.ToTensor(),
            ]
        self.aug = torchvision.transforms.Compose(tmp)

        self.sizeP = len(os.listdir(self.root + "/good"))
        self.sizeN = len(os.listdir(self.root + "/bad"))

    def __len__(self):
        return self.sizeP + self.sizeN

    def __getitem__(self, idx):
        if idx < self.sizeN:
            path = self.root + "/bad/" + str(idx) + ".pgn"
            label = 0
        else:
            path = self.root + "/good/" + str(idx - self.sizeN) + ".pgn"
            label = 1
        image = torchvision.io.read_image(path)

        size = image.shape[0] * image.shape[0] + image.shape[1] * image.shape[1]
        return self.transform(image), label, size
