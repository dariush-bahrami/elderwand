import math
from typing import Callable, NamedTuple, Sequence

from scipy.stats import logistic
from tqdm import tqdm


class TrainingMetrics(NamedTuple):
    discriminator_output_on_real_data: float
    discriminator_output_on_fake_data: float
    discriminator_loss: float
    generator_loss: float


class DiscriminatorSigmoidApplierHook:
    def __init__(self, epsilon: float = 1e-8):
        self.metrics_to_apply = [
            "discriminator_output_on_real_data",
            "discriminator_output_on_fake_data",
        ]
        self.sigmoid = lambda x: 1 / (1 + math.exp(-x) + epsilon)

    def __call__(self, mini_batch_metrics: TrainingMetrics) -> TrainingMetrics:
        updated_metrics = {}
        for key, value in mini_batch_metrics._asdict().items():
            if key not in self.metrics_to_apply:
                updated_metrics[key] = value
            else:
                updated_metrics[key] = self.sigmoid(value)
        return TrainingMetrics(**updated_metrics)


class MetricsAggregatorHook:
    def __init__(self):
        self.metrics = {
            "discriminator_output_on_real_data": [],
            "discriminator_output_on_fake_data": [],
            "discriminator_loss": [],
            "generator_loss": [],
        }

    def __call__(self, mini_batch_metrics: TrainingMetrics) -> TrainingMetrics:
        for key, value in mini_batch_metrics._asdict().items():
            self.metrics[key].append(value)
        return mini_batch_metrics


class TQDMHook:
    def __init__(self, progress_bar: tqdm):
        self.iteration = 0
        self.progress_bar = progress_bar
        self.name_mapper = {
            "discriminator_output_on_real_data": "D(x)",
            "discriminator_output_on_fake_data": "D(G(z))",
        }

    def __call__(self, mini_batch_metrics: TrainingMetrics) -> TrainingMetrics:
        self.iteration += 1
        postfix = []
        for key in self.name_mapper:
            name = self.name_mapper[key]
            value = mini_batch_metrics._asdict()[key]
            postfix.append(f"{name} = {value:.2f}")
        self.progress_bar.set_postfix_str(", ".join(postfix))
        return mini_batch_metrics


class ChainedHooks:
    def __init__(self, *hooks: Callable[[TrainingMetrics], None]):
        self.hooks: Sequence[str, Callable[[TrainingMetrics], None]] = hooks

    def __call__(self, mini_batch_metrics: TrainingMetrics) -> TrainingMetrics:
        for hook in self.hooks:
            mini_batch_metrics = hook(mini_batch_metrics)
