from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from ...trainer.applications.vision import VisionGANTrainer
from .discriminator import Discriminator
from .generator import Generator
from typing import Sequence
from torchvision import transforms as T
from ...data import ImageDataset


class DCGAN:
    def __init__(
        self,
        discriminator_feature_map_sizes: int,
        generator_feature_map_sizes: int,
        generator_latent_dim: int,
        image_channels: int,
        device: torch.device,
    ):
        self.device = device
        self.discriminator = Discriminator(
            image_channels,
            discriminator_feature_map_sizes,
        ).to(device)
        self.generator = Generator(
            generator_latent_dim,
            image_channels,
            generator_feature_map_sizes,
        ).to(device)

    def get_dataloader(
        self,
        image_paths: Sequence[Path],
        batch_size: int,
        num_dataloader_wrokers: int,
    ) -> DataLoader:
        image_size = 64
        transform = T.Compose(
            [
                T.Resize(image_size),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        dataset = ImageDataset(image_paths, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_dataloader_wrokers,
        )
        return dataloader

    def get_trainer(
        self,
        dataloader: DataLoader,
        tensorboard_log_dir: Path,
        image_sample_interval: int,
        image_sample_size: int,
        adam_optimizer_learning_rate=0.0002,
        adam_optimizer_learning_beta_1=0.5,
        adam_optimizer_learning_beta_2=0.999,
    ) -> VisionGANTrainer:
        lr = adam_optimizer_learning_rate
        beta1 = adam_optimizer_learning_beta_1
        beta2 = adam_optimizer_learning_beta_2
        generator_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=lr,
            betas=(beta1, beta2),
        )
        discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=lr,
            betas=(beta1, beta2),
        )
        criterion = nn.BCEWithLogitsLoss()
        return VisionGANTrainer(
            dataloader=dataloader,
            generator=self.generator,
            discriminator=self.discriminator,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            criterion=criterion,
            device=self.device,
            tensorboard_log_dir=tensorboard_log_dir,
            image_sample_interval=image_sample_interval,
            image_sample_size=image_sample_size,
        )
