import torch.nn as nn

from more_itertools import pairwise
from typing import Tuple, List
from dataclasses import dataclass, field

@dataclass
class SegmentTrainConfig:
    epochs: int
    learning_rate: float


@dataclass
class SegmentConfig:
    filters: List = field(default_factory=lambda: [32, 64, 32, 16])
    kernel_size: Tuple[int, int] = (3, 5)
    stride: Tuple[int, int] = (2, 1)
    padding: Tuple[int, int] = (0, 2)


class YellNet(nn.Module):
    def __init__(self, model_config: SegmentConfig):
        super().__init__()

        self.network = nn.Sequential()
        self._filters = [1] + model_config.filters
        for idx, (in_filter, out_filter) in enumerate(pairwise(self._filters)):
            self.network.add_module(f"conv_{idx}",
                nn.Conv2d(
                    in_filter,
                    out_filter,
                    model_config.kernel_size,
                    model_config.stride,
                    model_config.padding
                )
            )
            self.network.add_module(f"relu_{idx}", nn.LeakyReLU(0.1))
            self.network.add_module(f"dropout_{idx}", nn.Dropout(0.3))


        self.last_conv = nn.Conv2d(self._filters[-1], 1, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.network(x)
        x = self.last_conv(x)
        x = self.sigmoid(x)

        return x


