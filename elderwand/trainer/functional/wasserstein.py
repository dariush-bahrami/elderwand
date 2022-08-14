from typing import Callable, Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from ..metrics import TrainingMetrics


def _train_critic(
    critic: nn.Module,
    generator: nn.Module,
    critic_optimizer: optim.Optimizer,
    real_data: torch.Tensor,
    gradient_penalty_weight: float,
    device: torch.device,
) -> dict:
    critic.train()
    critic_optimizer.zero_grad()

    # Generate fake data
    batch_size = real_data.size(0)
    noise = generator.sample_noise(batch_size).to(device)
    fake_data = generator(noise).detach()

    # Compute critic loss
    #   Calculate gradient norm w.r.t. the mixture of real and fake input data
    epsilon = torch.rand_like(real_data).to(device)
    mixed_data = epsilon * real_data.to(device) + (1 - epsilon) * fake_data
    mixed_data.requires_grad_(True)
    critic_output_mixed = critic(mixed_data)
    gradient = torch.autograd.grad(
        outputs=critic_output_mixed,
        inputs=mixed_data,
        grad_outputs=torch.ones_like(critic_output_mixed),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient_norm = torch.flatten(gradient, start_dim=1).norm(p=2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

    #   Calculate the wasserstein loss
    critic_output_real = critic(real_data)
    critic_output_fake = critic(fake_data)
    wasserstein_loss = (
        -torch.mean(critic_output_real)
        + torch.mean(critic_output_fake)
        + gradient_penalty_weight * gradient_penalty
    )

    # Backpropagate and optimize
    wasserstein_loss.backward()
    critic_optimizer.step()
    return {
        "loss": wasserstein_loss.item(),
        "critic_output_on_real_data": critic_output_real.mean().item(),
        "critic_output_on_fake_data": critic_output_fake.mean().item(),
        "critic_gradient_norm": gradient_norm.mean().item(),
    }


def _train_generator(
    critic: nn.Module,
    generator: nn.Module,
    generator_optimizer: optim.Optimizer,
    batch_size: int,
    device: torch.device,
) -> dict:
    generator.train()
    generator_optimizer.zero_grad()

    # Generate fake data
    noise = generator.sample_noise(batch_size).to(device)
    fake_data = generator(noise)

    # Calculate the generator loss
    critic_output_fake = critic(fake_data)
    generator_loss = -torch.mean(critic_output_fake)

    # Backpropagate and optimize
    generator_loss.backward()
    generator_optimizer.step()

    return {
        "loss": generator_loss.item(),
        "critic_output_on_fake_data": critic_output_fake.mean().item(),
    }


METRICS = [
    "critic_output_on_real_data",
    "critic_output_on_fake_data",
    "critic_loss",
    "generator_loss",
    "critic_gradient_norm",
]


def train_one_epoch(
    dataloader: DataLoader,
    generator: nn.Module,
    critic: nn.Module,
    generator_optimizer: optim.Optimizer,
    critic_optimizer: optim.Optimizer,
    critic_train_per_batch: int,
    generator_train_per_batch: int,
    gradient_penalty_weight: float,
    device: torch.device,
    hook: Optional[Callable[[TrainingMetrics], TrainingMetrics]] = None,
) -> TrainingMetrics:
    """Train one epoch of the GAN model on the given data loader and return the metrics.

    Args:
        dataloader (DataLoader): Dataloader for the training data.
        critic (nn.Module): The critic model.
        generator (nn.Module): The generator model.
        critic_optimizer (optim.Optimizer): The optimizer for the critic.
        generator_optimizer (optim.Optimizer): The optimizer for the generator.
        device (torch.device): The device to use for training.
        hook (Optional[Callable[[TrainingMetrics], TrainingMetrics]], optional): A hook
            to call after each batch with the metric as arguments. Defaults to None.

    Returns:
        list[TrainingMetrics]: The list of metrics for each batch.
    """
    generator.train()
    critic.train()

    metrics = TrainingMetrics(metric_names=METRICS)
    for real_data in dataloader:
        real_data = real_data.to(device)
        batch_size = real_data.size(0)

        # Train the critic
        for _ in range(critic_train_per_batch):
            batch_critic_metrics = _train_critic(
                critic,
                generator,
                critic_optimizer,
                real_data,
                gradient_penalty_weight,
                device,
            )
            metrics.append_metric(
                "critic_output_on_real_data",
                batch_critic_metrics["critic_output_on_real_data"],
            )
            metrics.append_metric(
                "critic_output_on_fake_data",
                batch_critic_metrics["critic_output_on_fake_data"],
            )
            metrics.append_metric("critic_loss", batch_critic_metrics["loss"])
            metrics.append_metric(
                "critic_gradient_norm", batch_critic_metrics["critic_gradient_norm"]
            )

        # Train generator
        for _ in range(generator_train_per_batch):
            batch_generator_metrics = _train_generator(
                critic,
                generator,
                generator_optimizer,
                batch_size,
                device,
            )
            batch_generator_metrics["loss"]
            metrics.append_metric("generator_loss", batch_generator_metrics["loss"])

        if hook is not None:
            metrics = hook(metrics)
    return metrics
