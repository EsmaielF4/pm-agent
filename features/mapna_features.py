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

@register("feature_extractor", "mapna_sensor_columns_rolling")
class MAPNARollingFeatures(FeatureExtractor):
    """Same raw sensor columns as MAPNASensorColumnFeatures, PLUS rolling
    trend features per sensor: a moving average and moving standard
    deviation over the last `window` readings.

    Why this matters here specifically: the raw-value features only let
    the model ask "is this reading high or low right now?" They can't
    distinguish a sensor that's always run at, say, 60C from one that
    just climbed from 45C to 60C over the last few minutes - and a
    climbing trend is often the actual early-warning sign of a
    developing fault, not the instantaneous value. The rolling mean
    captures the recent trend level; the rolling std captures how
    noisy/unstable a sensor has become, which is itself a common
    precursor to failure (e.g. vibration getting erratic before a
    bearing fully fails).

    Requires raw_batch to be in chronological order (true for data
    from MAPNAMultiSensorSource) since rolling windows only look
    backward in time - never at future readings - to avoid leaking
    information a real live agent wouldn't have yet.
    """

    def __init__(self, sensor_columns: list, window: int = 5):
        self._cols = list(sensor_columns)
        self._window = window

    def transform(self, raw_batch: pd.DataFrame) -> pd.DataFrame:
        base = raw_batch[self._cols].copy()
        base = base.ffill().bfill()
        base = base.fillna(base.median(numeric_only=True))

        X = base.copy()
        for col in self._cols:
            roll = base[col].rolling(window=self._window, min_periods=1)
            X[f"{col}_roll_mean"] = roll.mean()
            X[f"{col}_roll_std"] = roll.std().fillna(0.0)
        return X

    def feature_names(self) -> list:
        names = list(self._cols)
        for col in self._cols:
            names += [f"{col}_roll_mean", f"{col}_roll_std"]
        return names