import torch
import torch.nn as nn

class PolyActivation(nn.Module):
    """
    Custom activation layer that applies a polynomial function to the input.
    
    Args:
        coefs (list): List of coefficients for the polynomial function. Default is [0., 0., 0.].
    
    Returns:
        Tensor: Output tensor after applying the polynomial function.
    """
    
    def __init__(self, coefs=None):
        super(PolyActivation, self).__init__()
        if coefs is None:
            coefs = [0., 0., 0.]
        self.coefs = torch.tensor(coefs, dtype=torch.float32, requires_grad=False)
    
    def forward(self, inputs):
        powers = torch.arange(len(self.coefs), dtype=torch.float32, device=inputs.device)
        powers = powers.unsqueeze(0)
        inputs = inputs.unsqueeze(-1)
        result = torch.sum(self.coefs * (inputs ** powers), dim=-2)
        return result


class SquareActivation(PolyActivation):
    """
    Custom activation layer that applies a square function to the input.
    Inherits from PolyActivation and sets the coefficients to [1., 0., 0.] by default.
    
    Returns:
        Tensor: Output tensor after applying the square function.
    """
    
    def __init__(self):
        super(SquareActivation, self).__init__([1., 0., 0.])
