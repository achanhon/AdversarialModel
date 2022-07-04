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
        Tr = True
        Pa = "build/"
        raw = torchvision.transforms.ToTensor()
        self.alldata = torchvision.datasets.EuroSAT(root=Pa, download=Tr, transform=raw)

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


if __name__=="__main__":
    lol = EurosatSplit("train")
    dataloader = torch.utils.data.DataLoader(lol, batch_size=64, shuffle=True)

    for x, y in dataloader:
        if y.shape[0] != 64:
            print(x.shape)
            torchvision.utils.save_image(x, "/home/achanhon/Bureau/lol.png")


