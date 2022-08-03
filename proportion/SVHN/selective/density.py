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


def selectivelogitTOdensity(logit):
    entropy = torch.sum(-logit * torch.log(logit + 0.000001), dim=1)
    entropyIndex = [(entropy[i], i) for i in range(entropy.shape[0])]
    entropyIndex = sorted(entropyIndex)
    selectedIndex = [entropyIndex[i][1] for i in range(int(0.8 * entropy.shape[0]))]
    tmp = [logit[i] for i in selectedIndex]
    logit = tmp.stack(logit)

    softmaxdensity = torch.nn.functional.softmax(logit, dim=1)
    tmp = torch.nn.functional.relu(logit) + softmaxdensity
    total = tmp.sum(dim=1)
    total = torch.stack([total] * logit.shape[1], dim=1)

    estimatedensity = tmp / total
    return estimatedensity.sum(dim=0) / logit.shape[0]
