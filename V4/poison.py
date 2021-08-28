import torch
import eval_feature


def compute_poisonedmodel(
    batchprovider, proxymodel, net, trainsize, featuredim, nbclasses
):
    ##############################################################################
    # hacker modifies TRAIN data to make the model the more distant from proxy one#
    ##############################################################################

    X = torch.Tensor(trainsize, 3, 32, 32).cuda()
    Y = torch.Tensor(trainsize).cuda().long()

    net.classifier = proxymodel
    i = 0
    for x, y in batchprovider:
        x, y = x.cuda(), y.cuda()
        pgd = eval_feature.pgd_attack(net, x, y)
        lenx = x.shape[0]
        X[i : i + lenx] = pgd
        Y[i : i + lenx] = y
        i += lenx

    poisonnedDataset = torch.utils.data.TensorDataset(X, Y)
    poisonnedloader = torch.utils.data.DataLoader(
        poisonnedDataset, batch_size=64, shuffle=True, num_workers=2
    )

    print("training on those data lead to poisoned model")
    net.classifier = torch.nn.Identity()
    poisonedmodel = eval_feature.trainClassifierOnFrozenfeature(
        poisonnedloader, net, trainsize, featuredim, nbclasses
    )

    return (
        poisonedmodel,
        poisonnedDataset,
    )  # poisonnedDataset is exported for display/verification


def eval_robustness_poisonning(
    trainloader, testloader, net, trainsize, testsize, featuredim, nbclasses
):
    print("eval clean model")
    clean_model = eval_feature.trainClassifierOnFrozenfeature(
        trainloader, net, trainsize, featuredim, nbclasses
    )
    net.classifier = clean_model
    cleanaccuracy = eval_feature.compute_accuracy(testloader, net, testsize)
    robustecleanaccuracy = eval_feature.compute_robust_accuracy(
        testloader, net, testsize
    )
    print("clean accuracy", cleanaccuracy, "robust", robustecleanaccuracy)

    print("generate poisoned model")
    net.classifier = torch.nn.Identity()
    proxy = eval_feature.trainClassifierOnFrozenfeature(
        testloader, net, testsize, featuredim, nbclasses
    )
    poisonnedmodel,_ = compute_poisonedmodel(
        trainloader, proxy, net, trainsize, featuredim, nbclasses
    )

    print("eval poisoned model")
    net.classifier = poisonnedmodel
    poisonaccuracy = eval_feature.compute_accuracy(testloader, net, testsize)
    robustepoisonaccuracy = eval_feature.compute_robust_accuracy(
        testloader, net, testsize
    )
    print("poison accuracy", poisonaccuracy, "robust", robustepoisonaccuracy)

    return cleanaccuracy, robustecleanaccuracy, poisonaccuracy, robustepoisonaccuracy
