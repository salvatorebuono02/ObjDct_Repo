import torch
import torch.nn.functional as F

class SquareActivation(torch.nn.Module):
    def __init__(self):
        super(SquareActivation, self).__init__()

    def forward(self, input):
        return torch.pow(input, 2)
