"""
Feature layer interface.

A FeatureExtractor turns raw rows from a DataSource into the numeric
feature matrix a model needs. Keeping this separate from both the data
source and the model means you can add new engineered features (rolling
stats, ratios, domain-specific signals) without touching either.
"""

from abc import ABC, abstractmethod
import pandas as pd


class FeatureExtractor(ABC):
    @abstractmethod
    def transform(self, raw_batch: pd.DataFrame) -> pd.DataFrame:
        """Return a numeric feature DataFrame derived from raw_batch."""
        raise NotImplementedError

    @abstractmethod
    def feature_names(self) -> list:
        raise NotImplementedError
