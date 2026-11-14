"""
Preprocesses the full ACN-Data JSON file into a clean, sorted CSV file
for fast, low-memory loading across all baseline models.

Usage:
  python preprocess_acn.py --input data/acn_data_caltech_full.json --output data/acn_caltech_clean.csv
"""

import os
import json
import argparse
import pandas as pd


def preprocess_acn_json_to_csv(json_path: str, csv_out_path: str):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Input JSON file '{json_path}' not found.")

    print(f"Loading raw JSON from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    items = raw.get("_items", raw if isinstance(raw, list) else [])
    records = []

    for idx, s in enumerate(items):
        conn = s.get("connectionTime")
        disc = s.get("disconnectTime")
        kwh = s.get("kWhDelivered")

        if not conn or not disc or kwh is None:
            continue
        try:
            kwh_val = float(kwh)
        except (ValueError, TypeError):
            continue
        if kwh_val <= 0.0:
            continue

        records.append({
            "sessionID": str(s.get("sessionID", s.get("_id", idx))),
            "siteID": str(s.get("siteID", "caltech")),
            "stationID": str(s.get("stationID", "")),
            "connectionTime": conn,
            "disconnectTime": disc,
            "kWhDelivered": kwh_val,
            "maxRate": float(s.get("maxRate", 6.6))
        })

    df = pd.DataFrame(records)
    df["connectionTime"] = pd.to_datetime(df["connectionTime"], utc=True, format="mixed")
    df["disconnectTime"] = pd.to_datetime(df["disconnectTime"], utc=True, format="mixed")

    # Filter out invalid durations and sort chronologically
    df = df[df["disconnectTime"] > df["connectionTime"]].sort_values("connectionTime").reset_index(drop=True)

    os.makedirs(os.path.dirname(csv_out_path) or ".", exist_ok=True)
    df.to_csv(csv_out_path, index=False)
    print(f"Successfully preprocessed {len(df)} valid sessions -> {csv_out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess ACN-Data JSON into a fast CSV.")
    parser.add_argument("--input", type=str, default="data/caltech_data.json", help="Input JSON path.")
    parser.add_argument("--output", type=str, default="data/acn_caltech_clean.csv", help="Output CSV path.")
    args = parser.parse_args()
    preprocess_acn_json_to_csv(args.input, args.output)