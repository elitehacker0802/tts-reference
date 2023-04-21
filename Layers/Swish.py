

import torch


class Swish(torch.nn.Module):
    """
    Construct an Swish activation function for Conformer.
    """

    def forward(self, x):
        """
        Return Swish activation function.
        """
        return x * torch.sigmoid(x)
