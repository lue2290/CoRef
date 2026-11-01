import torch
from torch import nn

class RegionHead(nn.Module):
    def __init__(self, dimension: int, n_classes: int):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(dimension, 256), nn.LayerNorm(256),
                                    nn.GELU(), nn.Dropout(0.3))
        self.known = nn.Linear(256, n_classes)
        self.foreground = nn.Linear(256, 1)

    def forward(self, x):
        z = self.shared(x)
        return self.known(z), self.foreground(z).squeeze(1)
