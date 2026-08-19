"""
Data layer interface - the "Perceive" stage.

Any new data source (a real dataset you download, live sensor feed,
a specific piece of equipment) implements this interface and registers
itself. The rest of the agent only ever calls get_batch() and never
cares where the rows came from.
"""

from abc import ABC, abstractmethod
import pandas as pd


class DataSource(ABC):
    """Contract every data source plugin must satisfy."""

    @abstractmethod
    def get_batch(self, n: int = 50) -> pd.DataFrame:
        """Return the next batch of n readings as a DataFrame.

        Columns must include whatever raw sensor/process fields the
        feature extractors downstream expect (documented per-source).
        """
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> dict:
        """Return metadata about this source: column names, units, equipment type."""
        raise NotImplementedError
