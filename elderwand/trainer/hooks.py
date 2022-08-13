"""Training hooks

In this module, the common training hooks are defined.
"""
import math
from typing import Callable, NamedTuple

from tqdm import tqdm


class TrainingMetrics(NamedTuple):
    discriminator_output_on_real_data: float
    discriminator_output_on_fake_data: float
    discriminator_loss: float
    generator_loss: float


class DiscriminatorSigmoidApplierHook:
    """Apply sigmoid function to discriminator output

    When using BCEWithLogitsLoss the output of the discriminator is not outputed from
    the sigmoid function. This hook is used to apply the sigmoid function to the output
    of the discriminator.
    """

    def __init__(self, epsilon: float = 1e-8):
        """Class constructor

        Args:
            epsilon (float, optional): A small number to avoid division by zero.
                Defaults to 1e-8.
        """
        self._metrics_to_apply: list[str] = [
            "discriminator_output_on_real_data",
            "discriminator_output_on_fake_data",
        ]
        self._sigmoid: Callable = lambda x: 1 / (1 + math.exp(-x) + epsilon)

    def __call__(self, mini_batch_metrics: TrainingMetrics) -> TrainingMetrics:
        """Apply sigmoid function to discriminator output

        Args:
            mini_batch_metrics (TrainingMetrics): Metrics of the current mini-batch.

        Returns:
            TrainingMetrics: Updated metrics of the current mini-batch.
        """
        updated_metrics = {}
        for key, value in mini_batch_metrics._asdict().items():
            if key not in self._metrics_to_apply:
                updated_metrics[key] = value
            else:
                updated_metrics[key] = self._sigmoid(value)
        return TrainingMetrics(**updated_metrics)


class MetricsAggregatorHook:
    """Store metrics during training

    Attributes:
        metrics (dict): Dictionary of metrics.
    """

    def __init__(self):
        self.metrics: dict[str, list[float]] = {
            "discriminator_output_on_real_data": [],
            "discriminator_output_on_fake_data": [],
            "discriminator_loss": [],
            "generator_loss": [],
        }

    def __call__(self, mini_batch_metrics: TrainingMetrics) -> TrainingMetrics:
        """Store metrics and return them unchanged

        Args:
            mini_batch_metrics (TrainingMetrics): Metrics of the current mini-batch.

        Returns:
            TrainingMetrics: Unchanged metrics of the current mini-batch.
        """
        for key, value in mini_batch_metrics._asdict().items():
            self.metrics[key].append(value)
        return mini_batch_metrics


class TQDMHook:
    """A hook that updates the progress bar during training.

    Attributes:
        progress_bar (tqdm): Progress bar to update.
    """

    def __init__(self, progress_bar: tqdm):
        """Class constructor

        Args:
            progress_bar (tqdm): Progress bar to update.
        """
        self.progress_bar = progress_bar
        self._iteration = 0
        self._name_mapper = {
            "discriminator_output_on_real_data": "D(x)",
            "discriminator_output_on_fake_data": "D(G(z))",
        }

    def __call__(self, mini_batch_metrics: TrainingMetrics) -> TrainingMetrics:
        """Update progress bar and return unchanged metrics

        Args:
            mini_batch_metrics (TrainingMetrics): Metrics of the current mini-batch.

        Returns:
            TrainingMetrics: Unchanged metrics of the current mini-batch.
        """
        self._iteration += 1
        postfix = []
        for key in self._name_mapper:
            name = self._name_mapper[key]
            value = mini_batch_metrics._asdict()[key]
            postfix.append(f"{name} = {value:.2f}")
        self.progress_bar.set_postfix_str(", ".join(postfix))
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
