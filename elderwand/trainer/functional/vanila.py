from typing import Callable, Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from ..hooks import TrainingMetrics


def train_one_epoch(
    dataloader: DataLoader,
    generator: nn.Module,
    discriminator: nn.Module,
    generator_optimizer: optim.Optimizer,
    discriminator_optimizer: optim.Optimizer,
    device: torch.device,
    hook: Optional[Callable[[TrainingMetrics], TrainingMetrics]] = None,
) -> list[TrainingMetrics]:
    """Train one epoch of the GAN model on the given data loader and return the metrics.

    Args:
        dataloader (DataLoader): Dataloader for the training data.
        discriminator (nn.Module): The discriminator model.
        generator (nn.Module): The generator model.
        discriminator_optimizer (optim.Optimizer): The optimizer for the discriminator.
        generator_optimizer (optim.Optimizer): The optimizer for the generator.
        device (torch.device): The device to use for training.
        hook (Optional[Callable[[TrainingMetrics], TrainingMetrics]], optional): A hook
            to call after each batch with the metric as arguments. Defaults to None.

    Returns:
        list[TrainingMetrics]: The list of metrics for each batch.
    """
    generator.train()
    discriminator.train()

    real_label = 1
    fake_label = 0
    metrics = []
    for real_data in dataloader:
        real_data = real_data.to(device)
        batch_size = real_data.size(0)
        labels = torch.zeros((batch_size, 1), dtype=torch.float, device=device)

        # Train discriminator
        discriminator_optimizer.zero_grad()
        # Discriminator preferred labels for real_data images is 1
        labels.fill_(real_label)
        # d_x is the discriminator's output on the real_data images
        d_x = discriminator(real_data)
        loss_d_x = F.binary_cross_entropy_with_logits(d_x, labels)
        loss_d_x.backward()
        noise = generator.sample_noise(batch_size).to(device)
        fake = generator(noise)
        # Discriminator preferred labels for fake images is 0
        labels.fill_(fake_label)
        d_g_z_1 = discriminator(fake.detach())
        loss_d_g_z_1 = F.binary_cross_entropy_with_logits(d_g_z_1, labels)
        # Update discriminator weights
        loss_d_g_z_1.backward()
        discriminator_optimizer.step()

        # Train generator
        generator_optimizer.zero_grad()
        # Generator preferred labels for fake images is 1
        labels.fill_(real_label)
        d_g_z_2 = discriminator(fake)
        loss_g_z = F.binary_cross_entropy_with_logits(d_g_z_2, labels)
        loss_g_z.backward()
        generator_optimizer.step()

        mini_batch_metrics = TrainingMetrics(
            discriminator_output_on_real_data=d_x.mean().item(),
            discriminator_output_on_fake_data=d_g_z_1.mean().item(),
            discriminator_loss=loss_d_x.item() + loss_d_g_z_1.item(),
            generator_loss=loss_g_z.item(),
        )

        metrics.append(mini_batch_metrics)

        if hook is not None:
            hook(mini_batch_metrics)
    return metrics
