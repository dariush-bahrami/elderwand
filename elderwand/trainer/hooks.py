"""Training hooks

In this module, the common training hooks are defined.
"""
import math
from typing import Callable, Optional, Sequence

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .metrics import TrainingMetrics


class SigmoidApplierHook:
    """Apply sigmoid function to discriminator output

    When using BCEWithLogitsLoss the output of the discriminator is not outputed from
    the sigmoid function. This hook is used to apply the sigmoid function to the output
    of the discriminator.
    """

    def __init__(self, metrics_to_apply: Sequence[str], epsilon: float = 1e-8):
        """Class constructor

        Args:
            epsilon (float, optional): A small number to avoid division by zero.
                Defaults to 1e-8.
        """
        self._metrics_to_apply = metrics_to_apply
        self._sigmoid: Callable = lambda x: 1 / (1 + math.exp(-x) + epsilon)

    def __call__(self, mini_batch_metrics: TrainingMetrics) -> TrainingMetrics:
        """Apply sigmoid function to discriminator output

        Args:
            mini_batch_metrics (TrainingMetrics): Metrics of the current mini-batch.

        Returns:
            TrainingMetrics: Updated metrics of the current mini-batch.
        """
        for metric in self._metrics_to_apply:
            mini_batch_metrics.metrics[metric] = self._sigmoid(
                mini_batch_metrics.metrics[metric]
            )
        return mini_batch_metrics


class MetricsAggregatorHook:
    """Store metrics during training

    Attributes:
        metrics (dict): Dictionary of metrics.
    """

    def __init__(self, metric_names: Sequence[str]):
        self.metrics = {metric: [] for metric in metric_names}

    def __call__(self, mini_batch_metrics: TrainingMetrics) -> TrainingMetrics:
        """Store metrics and return them unchanged

        Args:
            mini_batch_metrics (TrainingMetrics): Metrics of the current mini-batch.

        Returns:
            TrainingMetrics: Unchanged metrics of the current mini-batch.
        """
        batch_average_metrics = mini_batch_metrics.average()
        for key in self.metrics:
            self.metrics[key].append(batch_average_metrics[key])
        return mini_batch_metrics


class TQDMHook:
    """A hook that updates the progress bar during training.

    Attributes:
        progress_bar (tqdm): Progress bar to update.
    """

    def __init__(
        self,
        progress_bar: tqdm,
        metrics_to_print: Sequence[str],
        name_mapper: Optional[dict] = None,
    ):
        """Class constructor

        Args:
            progress_bar (tqdm): Progress bar to update.
        """
        self.progress_bar = progress_bar
        self.metrics_to_print = metrics_to_print
        self.name_mapper = name_mapper if name_mapper else {}
        self._iteration = 0

    def __call__(self, mini_batch_metrics: TrainingMetrics) -> TrainingMetrics:
        """Update progress bar and return unchanged metrics

        Args:
            mini_batch_metrics (TrainingMetrics): Metrics of the current mini-batch.

        Returns:
            TrainingMetrics: Unchanged metrics of the current mini-batch.
        """
        self._iteration += 1
        postfix = []
        average_metrics = mini_batch_metrics.average()
        for key in self.metrics_to_print:
            name = self.name_mapper.get(key, key)
            value = average_metrics[key]
            postfix.append(f"{name} = {value:.2f}")
        self.progress_bar.set_postfix_str(", ".join(postfix))
        return mini_batch_metrics


class TensorBoardHook:
    def __init__(self, writer: SummaryWriter):
        self.iteration = 0
        self.writer = writer

    def __call__(self, mini_batch_metrics: TrainingMetrics) -> TrainingMetrics:
        self.iteration += 1
        average_metrics = mini_batch_metrics.average()
        for key, value in average_metrics.items():
            self.writer.add_scalar(key, value, self.iteration)
        return mini_batch_metrics


class ChainedHooks:
    """Chain multiple hooks

    This hook is used to chain multiple hooks together and call them in sequence. The
    hooks are called in the order they are passed to the constructor. The output of the
    each hook is passed to the next hook in the sequence and the output of the last hook
    is returned.

    Attributes:
        hooks (tuple): Hooks to chain.
    """

    def __init__(self, *hooks: Callable[[TrainingMetrics], TrainingMetrics]):
        """Class constructor

        Args:
            *hooks (Callable[[TrainingMetrics], TrainingMetrics]): Hooks to chain.
        """
        self.hooks: tuple[Callable[[TrainingMetrics], TrainingMetrics]] = hooks

    def __call__(self, mini_batch_metrics: TrainingMetrics) -> TrainingMetrics:
        """Call all hooks in sequence and return the output of the last hook

        Args:
            mini_batch_metrics (TrainingMetrics): Metrics of the current mini-batch.

        Returns:
            TrainingMetrics: Output of the last hook.
        """
        for hook in self.hooks:
            mini_batch_metrics = hook(mini_batch_metrics)
        return mini_batch_metrics
