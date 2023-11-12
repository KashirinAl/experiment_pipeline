import yaml
import config
import abc
import pandas as pd
import numpy as np
from yaml.loader import SafeLoader
from os import listdir


def _load_yaml_preset():
    preset_path = config.PATH_METRIC_CONFIGS
    metrics_to_load = listdir(preset_path)
    metrics = []
    for metric in metrics_to_load:
        with open(preset_path + "/" + metric) as f:
            metrics.append(yaml.load(f, Loader=SafeLoader))
    return metrics


class Metric:
    def __init__(self, metric_config: dict):
        self._config = metric_config

    @property
    def estimators(self) -> str:
        return self._config.get("estimators", "default_value")

    @property
    def lifts(self) -> str:
        return self._config.get("lifts", "default_value")

    @property
    def transmorms(self) -> str:
        return self._config.get("transforms", "default_value")
        
    @staticmethod
    def _map_aggregation_function(aggregation_function: str) -> callable:
        mappings = {
            "count_distinct": pd.Series.nunique,
            "sum": np.sum
        }
        if aggregation_function not in mappings:
            raise ValueError(f"{aggregation_function} not found in mappings")
        return mappings[aggregation_function]


class CalculateMetric:
    def __init__(self, metric: Metric):
        self.metric = metric

    def __call__(self, df):
        return df.groupby([config.VARIANT_COL, self.metric.level]).apply(
            lambda df: pd.Series({
                "num": self.metric.numerator_aggregation_function(df[self.metric.numerator_aggregation_field]),
                "den": self.metric.denominator_aggregation_function(df[self.metric.denominator_aggregation_field]),
                "n": pd.Series.nunique(df[self.metric.level])
            })
        ).reset_index()


