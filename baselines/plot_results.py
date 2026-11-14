"""
Aggregates and plots the simulation results across all three baseline models:
  1. Naive Plug-and-Charge (Uncontrolled / FIFS)
  2. Lee et al. (Deterministic TOU-Only MPC / LP)
  3. Calafiore et al. (Class-Based / Robust LP)

Generates a multi-panel comparison figure ('baseline_comparison_results.png') and
exports a summary CSV ('baseline_summary_metrics.csv') for the ECE 271A report.

Usage:
  python plot_results.py --acn_file data/acn_caltech_clean.csv [--sample_date 2018-09-05] [--max_days 30]
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from baseline_naive_plug_and_charge import (
    load_all_acn_sessions_df,
    build_sce_tou_tariff_vector,
    run_naive_full_horizon,
)
from baseline_lee_et_al import (
    get_sce_tou_ev4_daily_tariff,
    solve_lee_single_window_sparse,
    run_lee_full_dataset,
)
from baseline_calafiore_et_al import (
    solve_calafiore_cohort_lp_vectorized,
    run_calafiore_full_acn_dataset,
)


def extract_single_day_profiles(
    df: pd.DataFrame,
    sample_date_str: str = "2018-09-05",
    dt_minutes: int = 5,
    eta: float = 1.0,
    transformer_cap_kw: float = 150.0,
    n_classes: int = 30,
    P0_kw: float = 6.6,
    uncertainty_eps: float = 0.0,
) -> dict:
    """
    Runs all three baseline models on a single representative 24-hour day
    from the ACN dataset so their 5-minute aggregate power profiles (kW)
    and cumulative energy deliveries (kWh) can be overlaid on the same plot.
    """
    dt_hours = dt_minutes / 60.0
    T_day = int(24 * 60 // dt_minutes)
    c_tou_day = get_sce_tou_ev4_daily_tariff(T=T_day, dt_hours=dt_hours)

    df_copy = df.copy()
    df_copy["date"] = df_copy["connectionTime"].dt.floor("D")
    target_ts = pd.Timestamp(sample_date_str, tz="UTC")

    day_df = df_copy[df_copy["date"] == target_ts]
    if day_df.empty:
        # Fallback to the day with the highest session count if sample_date is not in slice
        most_active_day = df_copy["date"].value_counts().index[0]
        print(f"Sample date {sample_date_str} not found; using most active date {most_active_day.strftime('%Y-%m-%d')}.")
        target_ts = most_active_day
        day_df = df_copy[df_copy["date"] == target_ts]

    # 1. Naive Plug-and-Charge 24-hour profile
    naive_power_kw = np.zeros(T_day, dtype=np.float64)
    window_sessions = []
    arrivals_mat = np.zeros((n_classes + 1, T_day), dtype=np.float64)

    for _, row in day_df.iterrows():
        arr = int((row["connectionTime"] - target_ts).total_seconds() // (dt_minutes * 60))
        dep = int((row["disconnectTime"] - target_ts).total_seconds() // (dt_minutes * 60))
        arr_c = max(0, min(T_day - 1, arr))
        dep_c = max(arr_c + 1, min(T_day, dep))

        total_dur = max(1, dep - arr)
        clipped_dur = dep_c - arr_c
        e_req = float(row["kWhDelivered"]) * (clipped_dur / total_dur)
        p_max = float(row.get("maxRate", 6.6))

        # Populate Naive profile
        rem = e_req
        for t in range(arr_c, dep_c):
            if rem <= 1e-6:
                break
            p_step = min(p_max, rem / (eta * dt_hours))
            naive_power_kw[t] += p_step
            rem -= p_step * eta * dt_hours

        # Populate Lee et al. session list
        window_sessions.append({
            "arr": arr_c,
            "dep": dep_c,
            "e_req": e_req,
            "p_max": p_max,
        })

        # Populate Calafiore et al. class arrival matrix
        cls_idx = min(int(np.ceil(e_req / (dt_hours * P0_kw))), n_classes)
        cls_idx = max(1, cls_idx)
        arrivals_mat[cls_idx, arr_c] += 1.0

    # 2. Lee et al. 24-hour profile
    lee_res = solve_lee_single_window_sparse(
        window_sessions,
        c_tou_day,
        T=T_day,
        dt_hours=dt_hours,
        eta=eta,
        transformer_cap_kw=transformer_cap_kw,
    )
    lee_power_kw = lee_res["agg_power_kw"]

    # 3. Calafiore et al. 24-hour profile
    alpha_base = np.array([(n_classes + 1 - i) / (n_classes + 2) for i in range(n_classes + 1)], dtype=np.float64)
    alpha_mat = np.repeat((0.15 * alpha_base)[:, None], T_day, axis=1)
    pv_zero = np.zeros(T_day, dtype=np.float64)

    cal_res = solve_calafiore_cohort_lp_vectorized(
        pi_t=c_tou_day,
        arrivals_matrix=arrivals_mat,
        departures_alpha=alpha_mat,
        pv_forecast_kw=pv_zero,
        T=T_day,
        n_classes=n_classes,
        dt_hours=dt_hours,
        P0_kw=P0_kw,
        P_ntw_max_kw=transformer_cap_kw,
        uncertainty_eps=uncertainty_eps,
    )
    cal_power_kw = cal_res["grid_power_kw"]

    hours_axis = np.arange(T_day) * dt_hours
    return {
        "date_str": target_ts.strftime("%Y-%m-%d"),
        "num_sessions": len(day_df),
        "hours": hours_axis,
        "c_tou": c_tou_day,
        "naive_kw": naive_power_kw,
        "lee_kw": lee_power_kw,
        "calafiore_kw": cal_power_kw,
    }


def generate_aggregated_plots(
    naive_metrics: dict,
    lee_metrics: dict,
    cal_metrics: dict,
    day_profiles: dict,
    out_png: str = "baseline_comparison_results.png",
):
    """
    Creates a 2x2 publication-quality figure aggregating:
      (a) Representative 24-hour Aggregate Power Draw (kW) + TOU Price Curve
      (b) Cumulative Energy Delivered Over the 24-Hour Window (kWh)
      (c) Full-Dataset Cost Breakdown (TOU Energy Cost vs. Peak Demand Charges)
      (d) Full-Dataset Peak Power Draw Comparison (All-Time Peak vs. Avg Monthly Peak)
    """
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
    })

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=200)
    fig.suptitle(
        "EV Fleet Charging Baseline Comparison (Caltech ACN-Data & SCE TOU-EV-4)",
        fontweight="bold",
        y=0.98,
    )

    colors = {
        "naive": "#d95f02",      # Muted orange/red
        "lee": "#1b9e77",        # Muted teal/green
        "calafiore": "#7570b3",  # Muted purple/blue
        "tou": "#666666",        # Grey for tariff background
    }

    hours = day_profiles["hours"]
    dt_hours = hours[1] - hours[0] if len(hours) > 1 else 5.0 / 60.0

    # -------------------------------------------------------------------------
    # Panel (a): 24-Hour Aggregate Power Profile & TOU Tariff
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1_tou = ax1.twinx()

    ax1_tou.fill_between(
        hours,
        0,
        day_profiles["c_tou"],
        color=colors["tou"],
        alpha=0.13,
        step="post",
        label="SCE TOU-EV-4 ($/kWh)",
    )
    ax1_tou.set_ylabel("TOU Price ($/kWh)", color=colors["tou"])
    ax1_tou.tick_params(axis="y", labelcolor=colors["tou"])
    ax1_tou.set_ylim(0, 0.35)

    ax1.plot(hours, day_profiles["naive_kw"], label="Naive Plug-and-Charge", color=colors["naive"], lw=1.8)
    ax1.plot(hours, day_profiles["lee_kw"], label="Lee et al. (TOU LP)", color=colors["lee"], lw=1.8)
    ax1.plot(hours, day_profiles["calafiore_kw"], label="Calafiore et al. (Class LP)", color=colors["calafiore"], lw=1.8, ls="--")

    ax1.set_title(f"(a) 24-Hour Aggregate Power Profile ({day_profiles['date_str']}, N={day_profiles['num_sessions']} EVs)")
    ax1.set_xlabel("Hour of Day (UTC)")
    ax1.set_ylabel("Aggregate Station Power (kW)")
    ax1.set_xlim(0, 24)
    ax1.grid(True, linestyle=":", alpha=0.6)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_tou.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.9)

    # -------------------------------------------------------------------------
    # Panel (b): Cumulative Energy Delivered Over the Day (kWh)
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    cum_naive = np.cumsum(day_profiles["naive_kw"]) * dt_hours
    cum_lee = np.cumsum(day_profiles["lee_kw"]) * dt_hours
    cum_cal = np.cumsum(day_profiles["calafiore_kw"]) * dt_hours

    ax2.plot(hours, cum_naive, label=f"Naive ({cum_naive[-1]:.1f} kWh)", color=colors["naive"], lw=2.0)
    ax2.plot(hours, cum_lee, label=f"Lee et al. ({cum_lee[-1]:.1f} kWh)", color=colors["lee"], lw=2.0)
    ax2.plot(hours, cum_cal, label=f"Calafiore et al. ({cum_cal[-1]:.1f} kWh)", color=colors["calafiore"], lw=2.0, ls="--")

    ax2.set_title(f"(b) Cumulative Energy Delivered ({day_profiles['date_str']})")
    ax2.set_xlabel("Hour of Day (UTC)")
    ax2.set_ylabel("Cumulative Energy Delivered (kWh)")
    ax2.set_xlim(0, 24)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", framealpha=0.9)

    # -------------------------------------------------------------------------
    # Panel (c): Full-Horizon Operating Cost Breakdown (Stacked Bar)
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    models = ["Naive\nPlug & Charge", "Lee et al.\n(Deterministic LP)", "Calafiore et al.\n(Class LP)"]
    tou_costs = [
        naive_metrics["tou_energy_cost_usd"],
        lee_metrics["tou_energy_cost_usd"],
        cal_metrics["tou_energy_cost_usd"],
    ]
    demand_costs = [
        naive_metrics["total_demand_charge_usd"],
        lee_metrics["total_demand_charge_usd"],
        cal_metrics["total_demand_charge_usd"],
    ]

    x_pos = np.arange(len(models))
    width = 0.5

    b1 = ax3.bar(x_pos, tou_costs, width, label="TOU Energy Cost ($)", color="#386cb0")
    b2 = ax3.bar(x_pos, demand_costs, width, bottom=tou_costs, label="Peak Demand Charge ($)", color="#f0027f", alpha=0.85)

    for i in range(len(models)):
        total_c = tou_costs[i] + demand_costs[i]
        ax3.text(
            x_pos[i],
            total_c * 1.02,
            f"${total_c:,.0f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )

    ax3.set_title("(c) Cumulative Operating Cost Breakdown Across Dataset")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(models)
    ax3.set_ylabel("Cost ($ USD)")
    ax3.set_ylim(0, max(tou_costs[i] + demand_costs[i] for i in range(3)) * 1.15)
    ax3.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax3.legend(loc="upper right", framealpha=0.9)

    # -------------------------------------------------------------------------
    # Panel (d): Peak Facility Demand Comparison (Grouped Bar)
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    all_time_peaks = [
        naive_metrics["overall_peak_power_kw"],
        lee_metrics["overall_peak_power_kw"],
        cal_metrics["overall_peak_power_kw"],
    ]
    monthly_avg_peaks = [
        naive_metrics["avg_monthly_peak_kw"],
        lee_metrics["avg_monthly_peak_kw"],
        cal_metrics["avg_monthly_peak_kw"],
    ]

    w_grp = 0.35
    rects1 = ax4.bar(x_pos - w_grp / 2, all_time_peaks, w_grp, label="All-Time Peak Draw (kW)", color="#e41a1c", alpha=0.85)
    rects2 = ax4.bar(x_pos + w_grp / 2, monthly_avg_peaks, w_grp, label="Avg Monthly Peak Draw (kW)", color="#4daf4a", alpha=0.85)

    for rect in rects1:
        h = rect.get_height()
        ax4.text(rect.get_x() + rect.get_width() / 2.0, h * 1.01, f"{h:.1f}", ha="center", va="bottom", fontsize=8)
    for rect in rects2:
        h = rect.get_height()
        ax4.text(rect.get_x() + rect.get_width() / 2.0, h * 1.01, f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    ax4.set_title("(d) Peak Facility Power Draw Comparison")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(models)
    ax4.set_ylabel("Power (kW)")
    ax4.set_ylim(0, max(all_time_peaks) * 1.18 if max(all_time_peaks) > 0 else 100)
    ax4.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax4.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Saved aggregated comparison plot -> {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate and plot all three EV charging baseline models.")
    parser.add_argument("--acn_file", type=str, default="data/acn_caltech_clean.csv", help="Path to ACN CSV or JSON.")
    parser.add_argument("--sample_date", type=str, default="2018-09-05", help="Date (YYYY-MM-DD) for 24h profile overlay.")
    parser.add_argument("--transformer_cap", type=float, default=150.0, help="Station capacity limit in kW.")
    parser.add_argument("--uncertainty_eps", type=float, default=0.0, help="Calafiore interval uncertainty (e.g. 0.20).")
    parser.add_argument("--dt_minutes", type=int, default=15, help="Timestep duration in minutes (default: 5).")
    parser.add_argument("--max_days", type=int, default=None, help="Optional cap on days for faster testing.")
    parser.add_argument("--out_png", type=str, default="results/baseline_comparison_results.png", help="Output plot filename.")
    parser.add_argument("--out_csv", type=str, default="results/baseline_summary_metrics.csv", help="Output metrics CSV filename.")
    args = parser.parse_args()

    print(f"Loading ACN dataset from {args.acn_file}...")
    df = load_all_acn_sessions_df(args.acn_file)

    if args.max_days is not None:
        df["date"] = df["connectionTime"].dt.floor("D")
        keep_days = sorted(df["date"].unique())[:args.max_days]
        df = df[df["date"].isin(keep_days)].drop(columns=["date"]).reset_index(drop=True)
        print(f"Filtered to first {args.max_days} days ({len(df):,} sessions).")

    print("\n--- [1/4] Running Naive Plug-and-Charge Baseline ---")
    naive_res = run_naive_full_horizon(df, dt_minutes=args.dt_minutes)

    print("\n--- [2/4] Running Lee et al. Deterministic LP Baseline ---")
    lee_res = run_lee_full_dataset(
        df,
        dt_minutes=args.dt_minutes,
        transformer_cap_kw=args.transformer_cap,
        max_days=args.max_days,
    )

    print("\n--- [3/4] Running Calafiore et al. Class-Based LP Baseline ---")
    cal_res = run_calafiore_full_acn_dataset(
        df,
        dt_minutes=args.dt_minutes,
        station_cap_kw=args.transformer_cap,
        uncertainty_eps=args.uncertainty_eps,
        max_days=args.max_days,
    )

    print(f"\n--- [4/4] Extracting 24-Hour Overlay for {args.sample_date} & Plotting ---")
    day_profiles = extract_single_day_profiles(
        df,
        sample_date_str=args.sample_date,
        dt_minutes=args.dt_minutes,
        transformer_cap_kw=args.transformer_cap,
        uncertainty_eps=args.uncertainty_eps,
    )

    generate_aggregated_plots(naive_res, lee_res, cal_res, day_profiles, out_png=args.out_png)

    # Save summary table to CSV for easy inclusion in LaTeX / report
    summary_df = pd.DataFrame([
        {
            "Baseline Model": "Naive Plug-and-Charge",
            "Sessions Evaluated": naive_res["num_sessions"],
            "Total Energy Delivered (kWh)": round(naive_res["total_energy_delivered_kwh"], 2),
            "Total Shortfall (kWh)": round(naive_res["total_shortfall_kwh"], 2),
            "All-Time Peak (kW)": round(naive_res["overall_peak_power_kw"], 2),
            "Avg Monthly Peak (kW)": round(naive_res["avg_monthly_peak_kw"], 2),
            "TOU Energy Cost ($)": round(naive_res["tou_energy_cost_usd"], 2),
            "Demand Charges ($)": round(naive_res["total_demand_charge_usd"], 2),
            "Total Cost ($)": round(naive_res["total_cost_usd"], 2),
        },
        {
            "Baseline Model": "Lee et al. (Deterministic LP)",
            "Sessions Evaluated": lee_res["sessions_evaluated"],
            "Total Energy Delivered (kWh)": round(lee_res["total_energy_delivered_kwh"], 2),
            "Total Shortfall (kWh)": round(lee_res["total_shortfall_kwh"], 2),
            "All-Time Peak (kW)": round(lee_res["overall_peak_power_kw"], 2),
            "Avg Monthly Peak (kW)": round(lee_res["avg_monthly_peak_kw"], 2),
            "TOU Energy Cost ($)": round(lee_res["tou_energy_cost_usd"], 2),
            "Demand Charges ($)": round(lee_res["total_demand_charge_usd"], 2),
            "Total Cost ($)": round(lee_res["total_cost_usd"], 2),
        },
        {
            "Baseline Model": "Calafiore et al. (Class LP)",
            "Sessions Evaluated": cal_res["sessions_evaluated"],
            "Total Energy Delivered (kWh)": round(cal_res["total_energy_delivered_kwh"], 2),
            "Total Shortfall (kWh)": round(cal_res["total_shortfall_kwh"], 2),
            "All-Time Peak (kW)": round(cal_res["overall_peak_power_kw"], 2),
            "Avg Monthly Peak (kW)": round(cal_res["avg_monthly_peak_kw"], 2),
            "TOU Energy Cost ($)": round(cal_res["tou_energy_cost_usd"], 2),
            "Demand Charges ($)": round(cal_res["total_demand_charge_usd"], 2),
            "Total Cost ($)": round(cal_res["total_cost_usd"], 2),
        },
    ])
    summary_df.to_csv(args.out_csv, index=False)
    print(f"Saved summary metrics table -> {args.out_csv}\n")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
