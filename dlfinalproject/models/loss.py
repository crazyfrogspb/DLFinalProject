import torch
import torch.nn.functional as F


class HingeLoss(torch.nn.Module):
    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight

    def forward(self, pos, neg):
        return self.weight * (0.5 * torch.mean(F.relu(1. - pos)) + 0.5 * torch.mean(F.relu(1. + neg)))


class GeneratorLoss(torch.nn.Module):
    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight

    def forward(self, neg):
        return - self.weight * torch.mean(neg)


class L1Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, imgs, refined_imgs, coarse_imgs):
        return torch.mean(torch.abs(imgs - refined_imgs)) + torch.mean(torch.abs(imgs - coarse_imgs))
