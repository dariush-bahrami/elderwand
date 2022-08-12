import torch
from torch import nn

from .shared import dcgan_weights_init


class ConvTransposeBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequential(x)


class Generator(nn.Module):
    def __init__(self, latent_dim: int, image_channels: int, feature_map_sizes: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_channels = image_channels
        self.feature_map_sizes = feature_map_sizes
        self.sequential = nn.Sequential(
            ConvTransposeBlock(latent_dim, feature_map_sizes * 8, 4, 1, 0),
            ConvTransposeBlock(feature_map_sizes * 8, feature_map_sizes * 4, 4, 2, 1),
            ConvTransposeBlock(feature_map_sizes * 4, feature_map_sizes * 2, 4, 2, 1),
            ConvTransposeBlock(feature_map_sizes * 2, feature_map_sizes, 4, 2, 1),
            nn.ConvTranspose2d(feature_map_sizes, image_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        self.apply(dcgan_weights_init)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.sequential(input)

    def sample_noise(self, batch_size: int) -> torch.Tensor:
        return torch.randn(batch_size, self.latent_dim, 1, 1)

    @torch.no_grad()
    def generate(self, batch_size: int) -> torch.Tensor:
        self.eval()
        return self(self.sample_noise(batch_size))
