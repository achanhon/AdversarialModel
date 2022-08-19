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


import matplotlib.pyplot as plt

if __name__ == "__main__":
    for i in range(49484):
        if not os.path.exists("/home/achanhon/github/AdversarialModel/proportion/MDVD/selectivesearch/build/MDVD/train/bad/"+str(i)+".png"):
            print(i)
    for i in range(7039):
        if not os.path.exists("/home/achanhon/github/AdversarialModel/proportion/MDVD/selectivesearch/build/MDVD/train/good/"+str(i)+".png"):
            print(i)
    
    
    rawdata = MDVD("train")
    dataloader = torch.utils.data.DataLoader(rawdata, batch_size=256, shuffle=True)

    for x, y, s in dataloader:

        values = []
        s = torch.sqrt(s)
        for i in range(s.shape[0]):
            if y[i] == 1:
                values.append(s)

        print(values)

        s.hist(x, bins=200)
        plt.show()

        density = labelsT0density(y, s)
        print(density)
