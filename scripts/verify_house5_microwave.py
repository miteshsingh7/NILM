"""Forensic verification of House 5 microwave active count discrepancy."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add repo root to path
repo_root = Path("/Users/miteshsingh/Documents/projects/NILM")
sys.path.insert(0, str(repo_root))

from src.config import DEFAULT_APPLIANCE_THRESHOLDS

def main():
    print("=" * 80)
    print("SECTION 1 & 2: SOURCE DATA AUDIT FOR REDD HOUSE 5 MICROWAVE")
    print("=" * 80)

    fpath = repo_root / "data/processed/redd_real_house_5.csv"
    if not fpath.exists():
        print(f"Error: {fpath} does not exist!")
        return

    df = pd.read_csv(fpath, index_col=0, parse_dates=True)
    print(f"Dataset File: {fpath}")
    print(f"Total Rows: {len(df):,}")
    print(f"Start Timestamp: {df.index[0]}")
    print(f"End Timestamp:   {df.index[-1]}")
    duration_days = (df.index[-1] - df.index[0]).total_seconds() / 86400.0
    print(f"Time Span: {duration_days:.2f} days")
    print(f"Columns present: {list(df.columns)}")

    # Check 24h split
    day1_samples = 14400 # 24h at 6s
    df_cal = df.iloc[:day1_samples]
    df_eval = df.iloc[day1_samples:]
    print(f"Calibration Slice [0:14400]: {len(df_cal):,} rows ({df_cal.index[0]} to {df_cal.index[-1]})")
    print(f"Evaluation Slice  [14400:]:  {len(df_eval):,} rows ({df_eval.index[0]} to {df_eval.index[-1]})")

    if "microwave" not in df.columns:
        print("Microwave column not found in House 5!")
        return

    mw_all = df["microwave"].dropna()
    mw_cal = df_cal["microwave"].dropna()
    mw_eval = df_eval["microwave"].dropna()

    print("\n--- Power Distribution (Full Series, non-null) ---")
    print(f"Count (non-null): {len(mw_all):,}")
    print(f"Min:    {mw_all.min():.4f} W")
    print(f"Median: {mw_all.median():.4f} W")
    print(f"Mean:   {mw_all.mean():.4f} W")
    print(f"p75:    {np.percentile(mw_all, 75):.4f} W")
    print(f"p90:    {np.percentile(mw_all, 90):.4f} W")
    print(f"p95:    {np.percentile(mw_all, 95):.4f} W")
    print(f"p99:    {np.percentile(mw_all, 99):.4f} W")
    print(f"p99.9:  {np.percentile(mw_all, 99.9):.4f} W")
    print(f"Max:    {mw_all.max():.4f} W")

    print("\n--- Active Count Scan by Power Threshold ---")
    thresholds = [10.0, 15.0, 20.0, 50.0, 100.0, 150.0, 200.0]
    print(f"{'Threshold (W)':<15} | {'Total > th':<12} | {'Calib [0:24h] > th':<20} | {'Eval [24h:] > th':<18}")
    print("-" * 75)
    for th in thresholds:
        tot_gt = int(np.sum(mw_all > th))
        cal_gt = int(np.sum(mw_cal > th))
        eval_gt = int(np.sum(mw_eval > th))
        mark = " <-- Canonical Threshold" if th == 200.0 else (" <-- Used in buggy script" if th == 15.0 else "")
        print(f"> {th:<13.1f} | {tot_gt:<12,d} | {cal_gt:<20,d} | {eval_gt:<18,d}{mark}")

    print("\n--- Active Count Scan (>= th vs > th) ---")
    for th in [15.0, 200.0]:
        tot_ge = int(np.sum(mw_all >= th))
        tot_gt = int(np.sum(mw_all > th))
        eval_ge = int(np.sum(mw_eval >= th))
        eval_gt = int(np.sum(mw_eval > th))
        print(f"Threshold {th:.1f}W: >= total: {tot_ge}, > total: {tot_gt} | >= eval: {eval_ge}, > eval: {eval_gt}")

    print("\n--- Samples with Power >= 200.0W (Canonical Microwave Threshold) ---")
    active_samples = df[df["microwave"] >= 200.0]
    print(f"Total samples found >= 200.0W: {len(active_samples)}")
    for idx, (ts, row) in enumerate(active_samples.iterrows()):
        int_loc = df.index.get_loc(ts)
        is_cal = int_loc < day1_samples
        split_name = "Calibration [0:24h]" if is_cal else "Evaluation [24h:]"
        print(f"Sample #{idx+1}:")
        print(f"  Timestamp:         {ts}")
        print(f"  Integer Row Index: {int_loc:,} (of {len(df):,})")
        print(f"  Split:             {split_name}")
        print(f"  Microwave Power:   {row['microwave']:.2f} W")
        print(f"  Mains Power:       {row['mains']:.2f} W")
        other_apps = [c for c in ["fridge", "dishwasher", "washing_machine"] if c in row]
        for c in other_apps:
            print(f"  {c:<17s}: {row[c]:.2f} W")

    print("\n--- Standby / Clock Leakage Analysis around 15W-20W ---")
    standby_samples = df[(df["microwave"] >= 15.0) & (df["microwave"] < 50.0)]
    print(f"Total samples between 15W and 50W: {len(standby_samples):,}")
    print("Power values in this range (sample of unique rounded values):")
    print(standby_samples["microwave"].round(1).value_counts().head(10))

if __name__ == "__main__":
    main()
