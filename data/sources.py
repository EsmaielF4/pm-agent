"""
Concrete DataSource plugins.

- SyntheticAI4ISource: generates data matching the schema/logic of the
  well-known AI4I 2020 Predictive Maintenance dataset (milling machine:
  air/process temperature, rotational speed, torque, tool wear -> failure).
  Used now since the sandbox has no network access to download the real
  UCI file. The failure rules mirror the dataset's documented physics
  (heat dissipation failure, power failure, overstrain, tool wear failure)
  so the demo behaves realistically.

- CSVDataSource: drop-in loader for a real CSV (e.g. the actual AI4I file,
  or your own field data later). Same interface, so swapping data sources
  is a one-line config change, not a code change.
"""

import numpy as np
import pandas as pd

from core.registry import register
from data.base import DataSource


@register("data_source", "synthetic_ai4i")
class SyntheticAI4ISource(DataSource):
    """Generates rows shaped like the AI4I 2020 dataset, with realistic
    failure logic so predictions are meaningful, not random."""

    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)
        self._udi = 0

    def describe(self) -> dict:
        return {
            "name": "synthetic_ai4i",
            "equipment_type": "milling machine (synthetic, AI4I-2020 schema)",
            "columns": {
                "air_temperature_K": "Kelvin",
                "process_temperature_K": "Kelvin",
                "rotational_speed_rpm": "rpm",
                "torque_Nm": "Newton-meters",
                "tool_wear_min": "minutes",
                "type": "product quality variant: L/M/H",
            },
            "target": "machine_failure (0/1) with failure_mode label",
        }

    def get_batch(self, n: int = 50) -> pd.DataFrame:
        rng = self._rng
        air_temp = rng.normal(300, 2, n)
        process_temp = air_temp + rng.normal(10, 1, n)
        rot_speed = rng.normal(1500, 180, n).clip(min=800)
        torque = rng.normal(40, 10, n).clip(min=3)
        tool_wear = rng.uniform(0, 250, n)
        ptype = rng.choice(["L", "M", "H"], size=n, p=[0.6, 0.3, 0.1])

        power = torque * rot_speed * (2 * np.pi / 60)  # Watts
        temp_diff = process_temp - air_temp

        heat_diss_fail = (temp_diff < 8.6) & (rot_speed < 1380)
        power_fail = (power < 3500) | (power > 9000)
        overstrain_fail = (tool_wear * torque) > 11000
        toolwear_fail = tool_wear > 200

        failure = heat_diss_fail | power_fail | overstrain_fail | toolwear_fail
        # small random-noise failures, mirroring the dataset's RNF concept
        random_fail = rng.random(n) < 0.001
        failure = failure | random_fail

        mode = np.select(
            [heat_diss_fail, power_fail, overstrain_fail, toolwear_fail, random_fail],
            ["heat_dissipation", "power", "overstrain", "tool_wear", "random"],
            default="none",
        )

        udis = np.arange(self._udi, self._udi + n)
        self._udi += n

        df = pd.DataFrame(
            {
                "udi": udis,
                "type": ptype,
                "air_temperature_K": air_temp,
                "process_temperature_K": process_temp,
                "rotational_speed_rpm": rot_speed,
                "torque_Nm": torque,
                "tool_wear_min": tool_wear,
                "machine_failure": failure.astype(int),
                "failure_mode": mode,
            }
        )
        return df


@register("data_source", "csv_file")
class CSVDataSource(DataSource):
    """Loads real data from a CSV. Point this at the real AI4I file (or
    your own equipment data later) via config - no other code changes."""

    def __init__(self, path: str, target_col: str = "machine_failure"):
        self._df = pd.read_csv(path)
        self._target_col = target_col
        self._cursor = 0

    def describe(self) -> dict:
        return {
            "name": "csv_file",
            "path": "user-provided",
            "columns": list(self._df.columns),
            "target": self._target_col,
        }

    def get_batch(self, n: int = 50) -> pd.DataFrame:
        end = min(self._cursor + n, len(self._df))
        batch = self._df.iloc[self._cursor:end].copy()
        self._cursor = end if end < len(self._df) else 0  # loop for demo
        return batch
