"""
Feature extractor for the aligned MAPNA multi-sensor table: just the raw
sensor columns, with NaNs (from asof-merge gaps / genuine missing sensor
readings, both expected per the dataset's own documentation) forward/back
-filled then median-imputed. Deliberately simple for week 1 - rolling
statistics (trend, rate-of-change per sensor) are the natural next
extension once this baseline is proven, and are exactly the kind of
addition this architecture is built to make easy.
"""

import pandas as pd

from core.registry import register
from features.base import FeatureExtractor


@register("feature_extractor", "mapna_sensor_columns")
class MAPNASensorColumnFeatures(FeatureExtractor):
    def __init__(self, sensor_columns: list):
        self._cols = list(sensor_columns)

    def transform(self, raw_batch: pd.DataFrame) -> pd.DataFrame:
        X = raw_batch[self._cols].copy()
        X = X.ffill().bfill()
        X = X.fillna(X.median(numeric_only=True))
        return X

    def feature_names(self) -> list:
        return list(self._cols)
