from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from .. import functional as functional_trainers
from .. import hooks
from ..metrics import TrainingMetrics


class TensorBoardVisionHook(hooks.TensorBoardHook):
    def __init__(
        self,
        writer: SummaryWriter,
        generator: nn.Module,
        fixed_noise: torch.Tensor,
        image_sample_interval: int,
    ):
        self.generator = generator
        self.fixed_noise = fixed_noise
        self.image_sample_interval = image_sample_interval
        self._iteration = 0
        super().__init__(writer)

    def __call__(self, mini_batch_metrics: TrainingMetrics):
        if self._iteration % self.image_sample_interval == 0:
            with torch.no_grad():
                fake = self.generator(self.fixed_noise).cpu()
            fake = make_grid(
                fake,
                nrow=int(self.fixed_noise.shape[0] ** 0.5),
                padding=2,
                normalize=True,
            )
            self.writer.add_image("generated_images", fake, self.iteration)
        self._iteration += 1
        return super().__call__(mini_batch_metrics)


class WGANTrainer:
    def __init__(
        self,
        dataloader: DataLoader,
        generator: nn.Module,
        critic: nn.Module,
        generator_optimizer: optim.Optimizer,
        critic_optimizer: optim.Optimizer,
        device: torch.device,
        critic_train_per_batch: int,
        generator_train_per_batch: int,
        gradient_penalty_weight: float,
        tensorboard_log_dir: Path,
        image_sample_interval: int,
        image_sample_size: int,
    ):
        self.dataloader = dataloader
        self.critic = critic
        self.generator = generator
        self.critic_optimizer = critic_optimizer
        self.generator_optimizer = generator_optimizer
        self.device = device
        self.critic_train_per_batch = critic_train_per_batch
        self.generator_train_per_batch = generator_train_per_batch
        self.gradient_penalty_weight = gradient_penalty_weight
        self.metrics_aggregator_hook = hooks.MetricsAggregatorHook(
            functional_trainers.wasserstein.METRICS
        )
        self.tensorboard_hook = TensorBoardVisionHook(
            SummaryWriter(tensorboard_log_dir),
            self.generator,
            self.generator.sample_noise(image_sample_size).to(device),
            image_sample_interval,
        )
        self.tqdm_metrics_to_print = ["critic_loss", "generator_loss"]

    @property
    def metrics(self):
        return self.metrics_aggregator_hook.metrics

    def train(self, epochs: int) -> None:
        progress_bar = tqdm(range(epochs), desc="Training")
        hook = hooks.ChainedHooks(
            self.metrics_aggregator_hook,
            hooks.TQDMHook(progress_bar, metrics_to_print=self.tqdm_metrics_to_print),
            self.tensorboard_hook,
        )
        for epoch in progress_bar:
            metrics = functional_trainers.wasserstein.train_one_epoch(
                self.dataloader,
                self.generator,
                self.critic,
                self.generator_optimizer,
                self.critic_optimizer,
                self.critic_train_per_batch,
                self.generator_train_per_batch,
                self.gradient_penalty_weight,
                self.device,
                hook,
            )


class VanilaTrainer:
    def __init__(
        self,
        dataloader: DataLoader,
        generator: nn.Module,
        discriminator: nn.Module,
        generator_optimizer: optim.Optimizer,
        discriminator_optimizer: optim.Optimizer,
        device: torch.device,
        tensorboard_log_dir: Path,
        image_sample_interval: int,
        image_sample_size: int,
    ):
        self.dataloader = dataloader
        self.discriminator = discriminator
        self.generator = generator
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer
        self.device = device
        self.metrics_aggregator_hook = hooks.MetricsAggregatorHook(
            functional_trainers.vanila.METRICS
        )
        self.tensorboard_hook = TensorBoardVisionHook(
            SummaryWriter(tensorboard_log_dir),
            self.generator,
            self.generator.sample_noise(image_sample_size).to(device),
            image_sample_interval,
        )
        self.tqdm_metrics_to_print = ["discriminator_loss", "generator_loss"]
        self.sigmoid_applier_hook = hooks.SigmoidApplierHook(
            [
                "discriminator_output_on_real_data",
                "discriminator_output_on_fake_data",
            ]
        )

    @property
    def metrics(self):
        return self.metrics_aggregator_hook.metrics

    def train(self, epochs: int) -> None:
        progress_bar = tqdm(range(epochs), desc="Training")
        hook = hooks.ChainedHooks(
            self.sigmoid_applier_hook,
            self.metrics_aggregator_hook,
            hooks.TQDMHook(progress_bar, metrics_to_print=self.tqdm_metrics_to_print),
            self.tensorboard_hook,
        )
        for epoch in progress_bar:
            metrics = functional_trainers.vanila.train_one_epoch(
                self.dataloader,
                self.generator,
                self.discriminator,
                self.generator_optimizer,
                self.discriminator_optimizer,
                self.device,
                hook,
            )
