import torch
import torchvision


def logitTOdensity(logit):
    softmaxdensity = torch.nn.functional.softmax(logit, dim=1)
    tmp = torch.nn.functional.relu(logit) + softmaxdensity
    total = tmp.sum(dim=1)
    total = torch.stack([total] * logit.shape[1], dim=1)

    estimatedensity = tmp / total
    return estimatedensity.sum(dim=0) / logit.shape[0]


def labelsTOdensity(targets, nbclasses=10):
    truedensity = torch.zeros(nbclasses).cuda()
    for i in range(nbclasses):
        truedensity[i] = (targets == i).float().sum() / targets.shape[0]
    return truedensity


def extendedKL(estimatedensity, truedensity):
    kl_loss = torch.nn.KLDivLoss(reduction="sum")
    kl = kl_loss(estimatedensity, truedensity)
    kl = torch.nn.functional.relu(kl)
    diff = estimatedensity - truedensity
    return kl + torch.sum(diff * diff) + diff.abs().sum()


class EurosatSplit(torch.utils.data.Dataset):
    def __init__(self, flag):
        assert flag in ["train", "test"]
        self.flag = flag

        Tr, Pa = True, "build/data"
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
        aug = torchvision.transforms.Compose(tmp)

        self.alldata = torchvision.datasets.EuroSAT(root=Pa, download=Tr, transform=aug)

        if self.flag == "train":
            self.size = len(self.alldata) * 2 // 3 - 2
        else:
            self.size = len(self.alldata) // 3 - 2

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.flag == "test":
            return self.alldata.__getitem__(idx * 3 + 2)
        else:
            lol = 3 * (idx // 2) + (idx % 2)  # thank oeis
            return self.alldata.__getitem__(lol)


def weightedlogitTOdensity(logit, weight):
    softmaxdensity = torch.nn.functional.softmax(logit, dim=1)
    tmp = torch.nn.functional.relu(logit) + softmaxdensity
    total = tmp.sum(dim=1)
    total = torch.stack([total] * logit.shape[1], dim=1)

    estimatedensity = tmp / total
    estimatedensity = estimatedensity / weight

    estimatedensity = estimatedensity.sum(dim=0)
    total = estimatedensity.sum(dim=0)
    return estimatedensity / total
