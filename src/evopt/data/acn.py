"""Sessions for one day of ACN-Data, read from a JSON file downloaded at
https://ev.caltech.edu/dataset (the same format the API returns).

"""
import json

import pandas as pd

from evopt.data.schema import hours_to_slot
from evopt.registry import SOURCES


@SOURCES.register("acn")
class ACNSessions:
    def __init__(self, path, date, ev_max_kw=6.6, timezone="America/Los_Angeles", claimed_only=True):
        self.path = path
        self.date = pd.Timestamp(date).date()
        self.ev_max_kw = ev_max_kw          # assumed charger rating; ACN-Data does not list it per session
        self.timezone = timezone
        self.claimed_only = claimed_only    # keep only sessions where the driver entered a request

    def load(self, site, rng) -> pd.DataFrame:
        with open(self.path) as f:
            items = json.load(f)["_items"]
        midnight = pd.Timestamp(self.date, tz=self.timezone)
        rows = []
        for item in items:
            inputs = item.get("userInputs") or []
            if self.claimed_only and not inputs:
                continue
            arrival = self._local(item["connectionTime"])
            if arrival.date() != self.date:
                continue
            departure = self._local(item["disconnectTime"])
            rows.append({
                "session_id": item["sessionID"],
                "driver_id": item.get("userID"),
                "arrival_hour": (arrival - midnight).total_seconds() / 3600,
                "departure_hour": (departure - midnight).total_seconds() / 3600,
                "energy_requested": inputs[-1]["kWhRequested"] if inputs else item["kWhDelivered"],
                "max_rate": self.ev_max_kw,
            })
        if not rows:
            raise ValueError(f"No ACN sessions found on {self.date} in {self.path}")
        df = pd.DataFrame(rows)
        df["arrival"] = hours_to_slot(df.pop("arrival_hour"), site)
        df["departure"] = hours_to_slot(df.pop("departure_hour"), site, round_up=True)
        return df

    def _local(self, timestamp: str) -> pd.Timestamp:
        return pd.to_datetime(timestamp, utc=True).tz_convert(self.timezone)
