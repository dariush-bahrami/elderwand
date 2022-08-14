from pathlib import Path
from typing import Sequence

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms as T

from ...data import ImageDataset
from ...trainer.applications.vision import WGANTrainer
from ..dcgan.discriminator import Discriminator as Critic
from ..dcgan.generator import Generator


class WDCGAN:
    def __init__(
        self,
        discriminator_feature_map_sizes: int,
        generator_feature_map_sizes: int,
        generator_latent_dim: int,
        image_channels: int,
        device: torch.device,
    ):
        self.device = device
        self.critic = Critic(
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
        image_sample_interval: int = 16,
        image_sample_size: int = 16,
        critic_train_per_batch: int = 1,
        generator_train_per_batch: int = 1,
        gradient_penalty_weight: float = 10.0,
        adam_optimizer_learning_rate=0.0002,
        adam_optimizer_learning_beta_1=0.5,
        adam_optimizer_learning_beta_2=0.999,
    ) -> WGANTrainer:
        lr = adam_optimizer_learning_rate
        beta1 = adam_optimizer_learning_beta_1
        beta2 = adam_optimizer_learning_beta_2
        generator_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=lr,
            betas=(beta1, beta2),
        )
        critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=lr,
            betas=(beta1, beta2),
        )
        return WGANTrainer(
            dataloader=dataloader,
            generator=self.generator,
            critic=self.critic,
            generator_optimizer=generator_optimizer,
            critic_optimizer=critic_optimizer,
            device=self.device,
            critic_train_per_batch=critic_train_per_batch,
            generator_train_per_batch=generator_train_per_batch,
            gradient_penalty_weight=gradient_penalty_weight,
            tensorboard_log_dir=tensorboard_log_dir,
            image_sample_interval=image_sample_interval,
            image_sample_size=image_sample_size,
        )
