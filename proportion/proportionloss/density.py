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


def kl_and_l2(estimatedensity, truedensity):
    kl = torch.nn.functional.kl_div(estimatedensity, truedensity)
    kl = torch.nn.functional.relu(kl)
    diff = estimatedensity - truedensity
    return kl + torch.sum(diff * diff)


def confusionmatrixTOdensity(cm):
    total = torch.sum(cm)
    accuracy = torch.sum(torch.diagonal(cm)) / total

    estimatedensity = cm.sum(dim=1) / total
    truedensity = cm.sum(dim=0) / total

    return (
        accuracy,
        kl_and_l2(estimatedensity, truedensity),
        estimatedensity,
        truedensity,
    )
