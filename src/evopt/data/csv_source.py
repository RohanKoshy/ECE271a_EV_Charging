"""Sessions from a CSV file, for whatever dataset we end up using.

Expected columns: arrival_hour, departure_hour (decimal hours since midnight),
energy_requested (kWh), max_rate (kW). Optional: session_id, driver_id,
p_safe (kW overcurrent limit).
"""
import numpy as np
import pandas as pd

from evopt.data.schema import hours_to_slot
from evopt.registry import SOURCES


@SOURCES.register("csv")
class CSVSessions:
    def __init__(self, path):
        self.path = path

    def load(self, site, rng) -> pd.DataFrame:
        df = pd.read_csv(self.path)
        df["arrival"] = hours_to_slot(df.pop("arrival_hour"), site)
        df["departure"] = hours_to_slot(df.pop("departure_hour"), site, round_up=True)
        if "session_id" not in df:
            df["session_id"] = np.arange(len(df))
        return df
