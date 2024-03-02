import torch
import torch.nn as nn


class SquareActivation(nn.Module):
    def forward(self, x):
        return torch.square(x)