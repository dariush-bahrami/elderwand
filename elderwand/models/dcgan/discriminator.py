import torch
from torch import nn
from .shared import dcgan_weights_init


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequential(x)


class Discriminator(nn.Module):
    def __init__(self, image_channels: int, feature_map_sizes: int) -> None:
        self.image_channels = image_channels
        self.feature_map_sizes = feature_map_sizes
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(image_channels, feature_map_sizes, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            ConvBlock(feature_map_sizes, feature_map_sizes * 2),
            ConvBlock(feature_map_sizes * 2, feature_map_sizes * 4),
            ConvBlock(feature_map_sizes * 4, feature_map_sizes * 8),
            nn.Conv2d(feature_map_sizes * 8, 1, 4, 1, 0, bias=False),
            nn.Flatten(),
            # There is no need for sigmoid in final layer (BCEWithLogitsLoss)
        )
        self.apply(dcgan_weights_init)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.sequential(input)
