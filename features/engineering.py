"""
Concrete FeatureExtractor plugins.
"""

import pandas as pd

from core.registry import register
from features.base import FeatureExtractor


@register("feature_extractor", "ai4i_basic")
class AI4IBasicFeatures(FeatureExtractor):
    """Raw sensor readings + a couple of physically-motivated derived
    features (temp differential, mechanical power, wear*torque strain
    proxy). Deliberately simple - this is the plugin you'd replace or
    extend with rolling windows / trend features once you have real
    time-series field data."""

    _cols = [
        "air_temperature_K",
        "process_temperature_K",
        "rotational_speed_rpm",
        "torque_Nm",
        "tool_wear_min",
        "temp_diff",
        "power_w",
        "strain_proxy",
    ]

    def transform(self, raw_batch: pd.DataFrame) -> pd.DataFrame:
        df = raw_batch.copy()
        df["temp_diff"] = df["process_temperature_K"] - df["air_temperature_K"]
        df["power_w"] = df["torque_Nm"] * df["rotational_speed_rpm"] * (
            2 * 3.14159265 / 60
        )
        df["strain_proxy"] = df["tool_wear_min"] * df["torque_Nm"]
        return df[self._cols]

    def feature_names(self) -> list:
        return list(self._cols)


@register("feature_extractor", "auto_numeric")
class AutoNumericFeatures(FeatureExtractor):
    """Picks up every numeric column in the batch automatically (minus a
    configurable exclude list, e.g. label columns) and median-imputes
    NaNs. Built for the MAPNA data: different problems expose different
    sensor sets (4 sensors for fault detection, 14 for multi-fault/RUL),
    so hardcoding a column list per problem would defeat the point of a
    reusable pipeline. This extractor adapts to whatever sensors are
    present - swap it out for a hand-crafted one once you know which
    engineered features (rolling stats, cross-sensor ratios) help most."""

    def __init__(self, exclude: list = None):
        self._exclude = set(exclude or [])
        self._medians = None
        self._cols = None

    def _numeric_cols(self, df: pd.DataFrame) -> list:
        return [
            c
            for c in df.columns
            if c not in self._exclude
            and c != "timestamp"
            and pd.api.types.is_numeric_dtype(df[c])
        ]

    def transform(self, raw_batch: pd.DataFrame) -> pd.DataFrame:
        cols = self._cols or self._numeric_cols(raw_batch)
        self._cols = cols
        X = raw_batch[cols].copy()
        if self._medians is None:
            self._medians = X.median(numeric_only=True)
        return X.fillna(self._medians)

    def feature_names(self) -> list:
        return list(self._cols or [])
