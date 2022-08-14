from typing import Callable, NamedTuple, Sequence


class InvalidMetricName(Exception):
    def __init__(self, metric_name: str):
        super().__init__(f"Invalid metric name: {metric_name}")


class TrainingMetrics:
    def __init__(self, metric_names: Sequence[str]):
        self.available_metrics = metric_names
        self.metrics = {metric: [] for metric in metric_names}

    def average(self) -> dict[str, float]:
        result = {}
        for key, value in self.metrics.items():
            result[key] = sum(value) / len(value)
        return result

    def latest(self) -> dict[str, float]:
        result = {}
        for key, value in self.metrics.items():
            result[key] = value[-1]
        return result

    def append_metric(self, metric_name: str, value: float):
        if metric_name not in self.available_metrics:
            raise InvalidMetricName(metric_name)
        else:
            self.metrics[metric_name].append(value)
