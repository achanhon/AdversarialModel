import torch


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


class BatchClassification(torch.nn.Module):
    def __init__(self, inputsize, outputsize):
        super(TwoHead, self).__init__()
        self.fc = torch.nn.Linear(inputsize*2, outputsize)

    def forward(self, x):
        xm,_ = x.max(dim=0)
        xa = x.mean(dim=0)
        x = torch.cat(xm,xa)
        return self.fc1(x)
