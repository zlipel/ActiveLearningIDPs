import torch
from torch import nn
import torch.nn.functional as F


class DNN(nn.Module):
    """Simple feed-forward neural network for regression tasks."""

    def __init__(self, dim_list, output_dim):
        """Initialize the network architecture.

        Parameters
        ----------
        dim_list : list[int]
            Sizes of hidden layers including the input dimension.
        output_dim : int
            Dimension of the network output.
        """
        super().__init__()
        self.linear = nn.ModuleList(
            [nn.Linear(dim_list[i], dim_list[i + 1]) for i in range(len(dim_list) - 1)]
        )
        self.out = nn.Linear(dim_list[-1], output_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        """Compute network output for a batch of features."""
        for layer in self.linear:
            x = self.activation(layer(x))
        return self.out(x)
