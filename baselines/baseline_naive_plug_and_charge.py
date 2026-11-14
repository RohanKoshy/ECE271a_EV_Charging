"""
Baseline 1: Naive Plug-and-Charge (Uncontrolled / FIFS) Model
Memory-Optimized for Full-Dataset Execution Across All Timesteps.

Instead of allocating an O(N * T) dense matrix (which requires ~450 GB of RAM for
30,000 sessions x 300,000 timesteps), this implementation streams sessions directly
into a 1D aggregate power vector of length T_total (requiring < 3 MB of RAM) and
computes exact monthly utility bills (TOU energy cost + monthly peak demand charges).
"""

import os
import json
import argparse
import numpy as np
import pandas as pd


def load_all_acn_sessions_df(filepath: str) -> pd.DataFrame:
    """Loads ACN sessions from a preprocessed CSV or raw JSON into a sorted DataFrame."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset file '{filepath}' not found. Run download_acn_batched.py and preprocess_acn.py first."
        )

    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
    elif filepath.endswith(".json"):
        with open(filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)
        items = raw.get("_items", raw if isinstance(raw, list) else [])
        rows = []
        for idx, s in enumerate(items):
            if s.get("connectionTime") and s.get("disconnectTime") and s.get("kWhDelivered"):
                rows.append({
                    "sessionID": s.get("sessionID", s.get("_id", idx)),
                    "connectionTime": s["connectionTime"],
                    "disconnectTime": s["disconnectTime"],
                    "kWhDelivered": float(s["kWhDelivered"]),
                    "maxRate": float(s.get("maxRate", 6.6))
                })
        df = pd.DataFrame(rows)
    else:
        raise ValueError("Provide a .csv or .json ACN dataset file.")

    df["connectionTime"] = pd.to_datetime(df["connectionTime"], utc=True, format="mixed")
    df["disconnectTime"] = pd.to_datetime(df["disconnectTime"], utc=True, format="mixed")
    df = df[(df["disconnectTime"] > df["connectionTime"]) & (df["kWhDelivered"] > 0)].copy()
    df.sort_values("connectionTime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def build_sce_tou_tariff_vector(time_index: pd.DatetimeIndex) -> np.ndarray:
    """
    Constructs the Southern California Edison (SCE) TOU-EV-4 tariff vector ($/kWh)
    across arbitrary timestamps (Lee et al., Table III).
    """
    hours = time_index.hour + time_index.minute / 60.0
    c_tou = np.full(len(time_index), 0.056, dtype=np.float64)
    mid_mask = ((hours >= 8.0) & (hours < 12.0)) | ((hours >= 18.0) & (hours < 23.0))
    on_mask = (hours >= 12.0) & (hours < 18.0)
    c_tou[mid_mask] = 0.092
    c_tou[on_mask] = 0.267
    return c_tou


def run_naive_full_horizon(
    df: pd.DataFrame,
    dt_minutes: int = 5,
    eta: float = 1.0,
    demand_charge_per_kw: float = 15.51
) -> dict:
    """
    Streams all sessions across the entire multi-year horizon in O(T_total) memory.
    """
    dt_hours = dt_minutes / 60.0
    sim_start = df["connectionTime"].min().floor(f"{dt_minutes}min")
    sim_end = df["disconnectTime"].max().ceil(f"{dt_minutes}min")

    total_steps = int((sim_end - sim_start).total_seconds() // (dt_minutes * 60))
    print(f"Horizon: {sim_start} to {sim_end} ({total_steps:,} steps of {dt_minutes} min)")

    # 1D aggregate power array uses < 5 MB of RAM even for 500,000 timesteps
    agg_power_kw = np.zeros(total_steps, dtype=np.float64)

    arr_steps = ((df["connectionTime"] - sim_start).dt.total_seconds() // (dt_minutes * 60)).to_numpy(dtype=np.int64)
    dep_steps = ((df["disconnectTime"] - sim_start).dt.total_seconds() // (dt_minutes * 60)).to_numpy(dtype=np.int64)
    e_reqs = df["kWhDelivered"].to_numpy(dtype=np.float64)
    p_maxs = df["maxRate"].to_numpy(dtype=np.float64) if "maxRate" in df.columns else np.full(len(df), 6.6)

    total_delivered_kwh = 0.0
    total_requested_kwh = float(np.sum(e_reqs))

    for i in range(len(df)):
        arr = max(0, int(arr_steps[i]))
        dep = min(total_steps, int(dep_steps[i]))
        if dep <= arr:
            continue

        e_req = e_reqs[i]
        p_max = p_maxs[i]
        energy_per_full_step = p_max * eta * dt_hours

        full_steps = int(e_req // energy_per_full_step)
        end_full = min(dep, arr + full_steps)

        if end_full > arr:
            agg_power_kw[arr:end_full] += p_max
            delivered = (end_full - arr) * energy_per_full_step
        else:
            delivered = 0.0

        rem = e_req - delivered
        if end_full < dep and rem > 1e-6:
            agg_power_kw[end_full] += rem / (eta * dt_hours)
            delivered += rem

        total_delivered_kwh += delivered

    # Build timestamp index to evaluate TOU energy costs and monthly peak demand charges
    time_index = pd.date_range(start=sim_start, periods=total_steps, freq=f"{dt_minutes}min", tz="UTC")
    c_tou = build_sce_tou_tariff_vector(time_index)

    tou_energy_cost_usd = float(np.sum(agg_power_kw * c_tou * dt_hours))
    overall_peak_kw = float(np.max(agg_power_kw)) if total_steps > 0 else 0.0

    # Group by Year-Month to compute commercial monthly peak demand charges
    ts_df = pd.DataFrame({"power_kw": agg_power_kw, "tou_cost": agg_power_kw * c_tou * dt_hours}, index=time_index)
    monthly_peaks = ts_df["power_kw"].resample("ME").max().fillna(0.0)
    total_demand_charge_usd = float(monthly_peaks.sum() * demand_charge_per_kw)

    return {
        "num_sessions": len(df),
        "total_steps": total_steps,
        "total_requested_kwh": total_requested_kwh,
        "total_energy_delivered_kwh": total_delivered_kwh,
        "total_shortfall_kwh": max(0.0, total_requested_kwh - total_delivered_kwh),
        "overall_peak_power_kw": overall_peak_kw,
        "avg_monthly_peak_kw": float(monthly_peaks.mean()),
        "tou_energy_cost_usd": tou_energy_cost_usd,
        "total_demand_charge_usd": total_demand_charge_usd,
        "total_cost_usd": tou_energy_cost_usd + total_demand_charge_usd
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory-Optimized Naive Baseline Across All Timesteps.")
    parser.add_argument("--acn_file", type=str, default="data/acn_caltech_clean.csv", help="Path to ACN CSV or JSON.")
    parser.add_argument("--dt_minutes", type=int, default=5, help="Timestep duration in minutes (default: 5).")
    args = parser.parse_args()

    sessions_df = load_all_acn_sessions_df(args.acn_file)
    res = run_naive_full_horizon(sessions_df, dt_minutes=args.dt_minutes)

    print("\n=== Naive Plug-and-Charge Baseline (Full Dataset) ===")
    print(f"Sessions Evaluated        : {res['num_sessions']:,}")
    print(f"Total Timesteps Simulated : {res['total_steps']:,}")
    print(f"Total Energy Delivered    : {res['total_energy_delivered_kwh']:,.2f} kWh")
    print(f"Total Energy Shortfall    : {res['total_shortfall_kwh']:,.2f} kWh")
    print(f"All-Time Peak Draw        : {res['overall_peak_power_kw']:,.2f} kW")
    print(f"Average Monthly Peak Draw : {res['avg_monthly_peak_kw']:,.2f} kW")
    print(f"Total TOU Energy Cost     : ${res['tou_energy_cost_usd']:,.2f}")
    print(f"Cumulative Demand Charges : ${res['total_demand_charge_usd']:,.2f}")
    print(f"Total Operating Cost      : ${res['total_cost_usd']:,.2f}")