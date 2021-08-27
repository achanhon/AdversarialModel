import torch
import eval_feature


def poison_dataset(
    batchprovider_train,
    batchprovider_test,
    encoder,
    trainsize,
    testsize,
    featuredim,
    nbclasses,
):
    epsilon = 3  # over 255

    print("hacker uses external data to find best possible classifier")
    bestpossibleclassifier = eval_feature.train_frozenfeature_classifier(
        batchprovider_train, encoder, trainsize, featuredim, nbclasses
    )

    print(
        "then, it modifies TRAIN data to make the model the more distant from this best possible model"
    )
    X = torch.Tensor(trainsize, 3, 32, 32).cuda()
    Y = torch.Tensor(trainsize).cuda().long()
