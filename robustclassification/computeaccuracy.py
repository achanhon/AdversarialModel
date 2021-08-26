import torch


def fsgm_attack(net, x, y, radius=3.0 / 255):
    criterion = torch.nn.CrossEntropyLoss()
    x.requires_grad = True
    opt = torch.optim.Adam([x], lr=1)

    loss = criterion(net(x), y)
    opt.zero_grad()
    loss.backward()

    adv_x = x + radius * x.grad.sign()
    return torch.clamp(adv_x, min=0, max=1)


def pgd_attack(net, x, y, radius=3.0 / 255, alpha=1.0 / 255, iters=40):
    criterion = torch.nn.CrossEntropyLoss()
    original_x = x

    for i in range(iters):
        x.requires_grad = True
        opt = torch.optim.Adam([x], lr=1)

        loss = criterion(net(x), y)
        opt.zero_grad()
        loss.backward()

        adv_x = x + alpha * x.grad.sign()
        eta = torch.clamp(adv_x - original_x, min=-radius, max=radius)
        x = (original_x + eta).detach_()
    return torch.clamp(x, min=0, max=1)


def accuracy(net, batchengine, radius=3.0 / 255, device="cuda"):
    clean_accuracy, robust_accuracy = [], []
    totallen = 0
    for x, y in batchengine:
        totallen += x.shape[0]
        x, y = x.to(device), y.to(device)

        z = net(x)
        clean_accuracy.append((z.max(1)[1] == y).float().sum())

        x_fsgm = fsgm_attack(net, x, y, radius)
        z_fsgm = net(x_fsgm)
        correct_fsgm = (z_fsgm.max(1)[1] == y).float()

        x_pgd = pgd_attack(net, x, y, radius)
        z_pgd = net(x_pgd)
        correct_pgd = (z_pgd.max(1)[1] == y).float()

        robust_accuracy.append(torch.min(correct_fsgm, correct_pgd).sum())

    clean_accuracy = torch.sum(torch.Tensor(clean_accuracy)) / totallen
    robust_accuracy = torch.sum(torch.Tensor(robust_accuracy)) / totallen
    return clean_accuracy.cpu().detach().numpy(), robust_accuracy.cpu().detach().numpy()


class MyFlatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


import os


def isonspiro():
    return os.path.exists("/scratchm/achanhon/thisfileisonspiro")
