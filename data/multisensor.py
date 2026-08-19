"""
Multi-sensor CSV data source for the MAPNA rotary-equipment dataset.

The core challenge this dataset introduces vs. the AI4I stand-in: sensors
are sampled at different rates (1s to 15s) and live in separate CSV files.
One sensor ("reference_sensor") carries the label at its own timestamps
(in train files); everything else has to be synchronized onto those
timestamps before it's usable as a feature matrix.

Two dataset shapes exist across the MAPNA problems, both handled here:

1. "unsynced" (P2, P4-style): each sensor is its own file with its own
   timestamp column, sampled at its own rate. Alignment uses
   pandas.merge_asof (nearest prior reading within a tolerance window)
   onto the reference sensor's timestamps.

2. "prealigned" (P3, P5-style): every sensor file already has the same
   number of rows in the same order (timestamp column stripped) - a
   simple column-wise concat is enough, no merge_asof needed.

Same DataSource interface as before, so it drops into the existing
Perceive -> Predict -> Decide -> Act loop unchanged; only the config
(config/agent_config.yaml) needs to point at it.
"""

import glob
import os

import pandas as pd

from core.registry import register
from data.base import DataSource


@register("data_source", "mapna_multisensor")
class MAPNAMultiSensorSource(DataSource):
    def __init__(
        self,
        data_dir: str,
        file_suffix: str = "_train.csv",
        reference_sensor: str = None,
        label_columns: list = None,
        already_aligned: bool = False,
        tolerance: str = "5s",
    ):
        """
        data_dir: folder containing one CSV per sensor, named
            "<SensorName>{file_suffix}", e.g. "Temperature_C_train.csv".
        reference_sensor: the sensor column whose timestamps/row-order
            define the alignment grid (usually the one carrying labels).
            If None, inferred as the sensor file containing label_columns.
        label_columns: column name(s) holding the target label(s), if
            present in this data_dir (omit/empty for unlabeled test data).
        already_aligned: True for pre-synced datasets (P3/P5-style) where
            every sensor file has identical row count/order and no
            timestamp column - skips merge_asof, does a column concat.
        tolerance: max time gap allowed when asof-matching a slower
            sensor's reading onto the reference timestamp (unsynced mode).
        """
        self.data_dir = data_dir
        self.file_suffix = file_suffix
        self.label_columns = label_columns or []
        self.already_aligned = already_aligned
        self.tolerance = pd.Timedelta(tolerance)

        files = sorted(glob.glob(os.path.join(data_dir, f"*{file_suffix}")))
        if not files:
            raise FileNotFoundError(f"No files matching *{file_suffix} in {data_dir}")

        self.sensor_names = [
            os.path.basename(f)[: -len(file_suffix)] for f in files
        ]

        if reference_sensor is None and self.label_columns:
            # auto-detect: the file that actually contains the label column(s)
            for f, name in zip(files, self.sensor_names):
                cols = pd.read_csv(f, nrows=0).columns
                if any(lc in cols for lc in self.label_columns):
                    reference_sensor = name
                    break
        self.reference_sensor = reference_sensor

        self._aligned = (
            self._build_aligned_prealigned(files)
            if already_aligned
            else self._build_aligned_unsynced(files)
        )
        self._cursor = 0

    # -- prealigned datasets (P3/P5-style: same row count, no timestamp) --
    def _build_aligned_prealigned(self, files) -> pd.DataFrame:
        frames = []
        label_frame_added = False
        for f, name in zip(files, self.sensor_names):
            df = pd.read_csv(f)
            value_col = [c for c in df.columns if c not in self.label_columns][0]
            frames.append(df[[value_col]].rename(columns={value_col: name}))
            if not label_frame_added:
                present = [c for c in self.label_columns if c in df.columns]
                if present:
                    frames.append(df[present])
                    label_frame_added = True
        return pd.concat(frames, axis=1)

    # -- unsynced datasets (P2/P4-style: per-sensor timestamps, merge_asof --
    def _build_aligned_unsynced(self, files) -> pd.DataFrame:
        ref_file = next(
            f for f, n in zip(files, self.sensor_names) if n == self.reference_sensor
        )
        ref_df = pd.read_csv(ref_file, parse_dates=["timestamp"]).sort_values("timestamp")
        value_col = [
            c for c in ref_df.columns if c not in ("timestamp", *self.label_columns)
        ][0]
        aligned = ref_df.rename(columns={value_col: self.reference_sensor})

        for f, name in zip(files, self.sensor_names):
            if name == self.reference_sensor:
                continue
            df = pd.read_csv(f, parse_dates=["timestamp"]).sort_values("timestamp")
            vcol = [c for c in df.columns if c != "timestamp"][0]
            df = df.rename(columns={vcol: name})[["timestamp", name]]
            aligned = pd.merge_asof(
                aligned, df, on="timestamp", direction="nearest", tolerance=self.tolerance
            )
        return aligned

    def describe(self) -> dict:
        return {
            "name": "mapna_multisensor",
            "data_dir": self.data_dir,
            "sensors": self.sensor_names,
            "reference_sensor": self.reference_sensor,
            "label_columns": self.label_columns,
            "n_rows": len(self._aligned),
            "already_aligned": self.already_aligned,
        }

    def get_batch(self, n: int = 50) -> pd.DataFrame:
        end = min(self._cursor + n, len(self._aligned))
        batch = self._aligned.iloc[self._cursor:end].copy()
        self._cursor = end if end < len(self._aligned) else 0
        return batch

    def full_data(self) -> pd.DataFrame:
        """Escape hatch for training on everything at once rather than in
        streaming batches - useful for train/test-split model fitting."""
        return self._aligned.copy()
