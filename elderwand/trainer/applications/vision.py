from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from ..functional import train_one_epoch
from .. import hooks


class TensorBoardHook:
    def __init__(
        self,
        writer: SummaryWriter,
        generator: nn.Module,
        fixed_noise: torch.Tensor,
        image_sample_interval: int,
    ):
        self.iteration = 0
        self.writer = writer
        self.generator = generator
        self.image_sample_interval = image_sample_interval
        self.fixed_noise = fixed_noise

    def __call__(self, mini_batch_metrics: hooks.TrainingMetrics):
        self.iteration += 1
        for key, value in mini_batch_metrics._asdict().items():
            self.writer.add_scalar(key, value, self.iteration)
        if self.iteration % self.image_sample_interval == 0:
            with torch.no_grad():
                fake = self.generator(self.fixed_noise).cpu()
            fake = make_grid(
                fake,
                nrow=int(self.fixed_noise.shape[0] ** 0.5),
                padding=2,
                normalize=True,
            )
            self.writer.add_image("generated_images", fake, self.iteration)


class VisionGANTrainer:
    def __init__(
        self,
        dataloader: DataLoader,
        generator: nn.Module,
        discriminator: nn.Module,
        generator_optimizer: optim.Optimizer,
        discriminator_optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        tensorboard_log_dir: Path,
        image_sample_interval: int,
        image_sample_size: int,
    ):
        self.dataloader = dataloader
        self.discriminator = discriminator
        self.generator = generator
        self.criterion = criterion
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer
        self.device = device
        self.metrics_aggregator_hook = hooks.MetricsAggregatorHook()
        self.tensorboard_hook = TensorBoardHook(
            SummaryWriter(tensorboard_log_dir),
            self.generator,
            self.generator.sample_noise(image_sample_size).to(device),
            image_sample_interval,
        )

    @property
    def metrics(self):
        return self.metrics_aggregator_hook.metrics

    def train(self, epochs: int) -> None:
        progress_bar = tqdm(range(epochs), desc="Training")
        hook = hooks.ChainedHooks(
            hooks.DiscriminatorSigmoidApplierHook(),
            self.metrics_aggregator_hook,
            hooks.TQDMHook(progress_bar),
            self.tensorboard_hook,
        )
        for epoch in progress_bar:
            metrics = train_one_epoch(
                self.dataloader,
                self.generator,
                self.discriminator,
                self.generator_optimizer,
                self.discriminator_optimizer,
                self.criterion,
                self.device,
                hook,
            )
