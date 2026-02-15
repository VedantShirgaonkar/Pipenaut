"""
Sharded MLP model for pipeline parallelism.

A simple multi-layer perceptron that is split across pipeline stages.
Each stage owns a slice of the layers. The last stage also holds the
loss function and classification head.
"""

import torch.nn as nn


class ShardedMLP(nn.Module):
    """
    A toy MLP sharded across multiple pipeline stages.

    Each stage gets `total_layers // world_size` hidden layers.
    The last stage additionally gets a classification head + loss function.
    """

    def __init__(self, dim: int, total_layers: int, rank: int, world_size: int):
        super().__init__()
        layers_per_stage = total_layers // world_size

        self.rank = rank
        self.is_first = rank == 0
        self.is_last = rank == world_size - 1

        layers = []
        for _ in range(layers_per_stage):
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.ReLU())

        if self.is_last:
            layers.append(nn.Linear(dim, 2))
            self.loss_fn = nn.CrossEntropyLoss()

        self.net = nn.Sequential(*layers)

    def forward(self, x, targets=None):
        """
        Forward pass through this stage's layers.

        On the last stage, if targets are provided, returns the loss scalar.
        Otherwise returns the hidden activations to send to the next stage.
        """
        x = self.net(x)

        if self.is_last and targets is not None:
            return self.loss_fn(x, targets)

        return x
