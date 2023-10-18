import torch


def entropy_loss(x):
    x1 = torch.reshape(x, (x.shape[0], -1))  # B, N
    probs = torch.div(x1.T, x1.sum(dim=-1)).T  # B, N
    entropy = -(probs * torch.clamp(torch.log(probs),
                min=torch.finfo(x.dtype).min)).sum(dim=-1)  # B
    return entropy.mean()


def kurtosis_loss(x):
    kur = x.pow(4).mean(dim=1) / x.pow(2).mean(dim=1).pow(2)  # B, T
    return kur  # .mean()


def kurtosis2_loss(x):
    kur = x.pow(4).mean() / x.pow(2).mean().pow(2)
    return kur  # .mean()
