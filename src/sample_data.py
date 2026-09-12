"""Realistic synthetic smart meter energy data generator.

Simulates realistic multi-appliance energy signatures and aggregate mains
at 6-second resolution. Useful for offline unit testing, end-to-end pipeline
verification, and running the Streamlit dashboard out of the box.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.config import DEFAULT_APPLIANCE_THRESHOLDS, DEFAULT_APPLIANCES


def generate_synthetic_house(
    num_days: int = 7,
    sample_period_seconds: int = 6,
    appliances: Optional[List[str]] = None,
    seed: int = 42,
    base_standby_watts: float = 80.0,
) -> pd.DataFrame:
    """Generates synthetic time series for aggregate mains and target appliances.

    Args:
        num_days: Duration of simulation in days.
        sample_period_seconds: Sampling frequency (default: 6 seconds).
        appliances: List of appliances to simulate.
        seed: Random seed for reproducibility.
        base_standby_watts: Always-on background power consumption.

    Returns:
        DataFrame indexed by datetime with columns: 'mains', 'fridge', etc.
    """
    appliances = appliances or DEFAULT_APPLIANCES
    rng = np.random.default_rng(seed)

    total_samples = int((num_days * 24 * 3600) / sample_period_seconds)
    timestamps = pd.date_range(
        start="2026-01-01 00:00:00",
        periods=total_samples,
        freq=f"{sample_period_seconds}s",
    )

    data: Dict[str, np.ndarray] = {
        app: np.zeros(total_samples, dtype=np.float32) for app in appliances
    }

    # 1. Fridge: Cyclic operation ~20-30 min on, ~40-60 min off
    if "fridge" in appliances:
        fridge_trace = np.zeros(total_samples, dtype=np.float32)
        idx = 0
        while idx < total_samples:
            off_duration = int(rng.integers(30, 60) * 60 / sample_period_seconds)
            on_duration = int(rng.integers(18, 30) * 60 / sample_period_seconds)
            idx += off_duration
            if idx >= total_samples:
                break
            end_on = min(idx + on_duration, total_samples)
            # Compressor startup spike (200-250W) decaying to running 100-130W
            run_len = end_on - idx
            fridge_cycle = 115.0 + rng.normal(0, 5, run_len)
            if run_len > 2:
                fridge_cycle[:2] += rng.uniform(80, 120, min(2, run_len))
            fridge_trace[idx:end_on] = np.clip(fridge_cycle, 0, None)
            idx = end_on
        data["fridge"] = fridge_trace

    # 2. Microwave: Short high-power bursts (800 - 1300W, 1 - 4 min)
    if "microwave" in appliances:
        mw_trace = np.zeros(total_samples, dtype=np.float32)
        # Average 3-5 events per day, concentrated in morning/evening
        num_events = int(num_days * rng.integers(3, 6))
        event_starts = rng.choice(np.arange(100, total_samples - 100), size=num_events, replace=False)
        for s in event_starts:
            duration = int(rng.integers(1, 4) * 60 / sample_period_seconds)
            power = rng.uniform(1050, 1300)
            end_s = min(s + duration, total_samples)
            mw_trace[s:end_s] = power + rng.normal(0, 15, end_s - s)
        data["microwave"] = np.clip(mw_trace, 0, None)

    # 3. Dishwasher: Multi-stage heating & wash cycles (1.5 - 2.5 hours)
    if "dishwasher" in appliances:
        dw_trace = np.zeros(total_samples, dtype=np.float32)
        # ~1 cycle every 1.5 days
        num_cycles = max(1, int(num_days * 0.7))
        cycle_starts = rng.choice(np.arange(500, total_samples - 2000), size=num_cycles, replace=False)
        for s in cycle_starts:
            # Stage 1: Pre-wash & heating (1600W, 20 min)
            t1 = int(20 * 60 / sample_period_seconds)
            # Stage 2: Agitation wash (150W, 35 min)
            t2 = int(35 * 60 / sample_period_seconds)
            # Stage 3: Rinse & main heat (1800W, 25 min)
            t3 = int(25 * 60 / sample_period_seconds)
            # Stage 4: Drying / circulate (80W, 20 min)
            t4 = int(20 * 60 / sample_period_seconds)

            cursor = s
            for duration, p_nom in [(t1, 1600.0), (t2, 150.0), (t3, 1800.0), (t4, 80.0)]:
                end_c = min(cursor + duration, total_samples)
                if cursor < total_samples:
                    dw_trace[cursor:end_c] = p_nom + rng.normal(0, 20, end_c - cursor)
                cursor = end_c
        data["dishwasher"] = np.clip(dw_trace, 0, None)

    # 4. Washing Machine: Wash agitation (250-400W) followed by spin cycles (800-1400W)
    if "washing_machine" in appliances:
        wm_trace = np.zeros(total_samples, dtype=np.float32)
        # ~1 cycle every 2 days
        num_cycles = max(1, int(num_days * 0.5))
        cycle_starts = rng.choice(np.arange(300, total_samples - 1500), size=num_cycles, replace=False)
        for s in cycle_starts:
            # Wash pulses for 40 min
            wash_len = int(40 * 60 / sample_period_seconds)
            end_w = min(s + wash_len, total_samples)
            pulses = (np.sin(np.linspace(0, 30 * np.pi, end_w - s)) > 0).astype(float) * 350.0
            wm_trace[s:end_w] = pulses + rng.normal(0, 20, end_w - s)

            # Spin cycle for 15 min
            spin_len = int(15 * 60 / sample_period_seconds)
            end_spin = min(end_w + spin_len, total_samples)
            if end_w < total_samples:
                wm_trace[end_w:end_spin] = 950.0 + rng.normal(0, 50, end_spin - end_w)
        data["washing_machine"] = np.clip(wm_trace, 0, None)

    # 5. Kettle (if present): ~2000W short bursts (2-4 min)
    if "kettle" in appliances:
        kt_trace = np.zeros(total_samples, dtype=np.float32)
        num_boils = int(num_days * rng.integers(4, 8))
        starts = rng.choice(np.arange(100, total_samples - 100), size=num_boils, replace=False)
        for s in starts:
            dur = int(rng.integers(2, 4) * 60 / sample_period_seconds)
            end_s = min(s + dur, total_samples)
            kt_trace[s:end_s] = 2100.0 + rng.normal(0, 30, end_s - s)
        data["kettle"] = np.clip(kt_trace, 0, None)

    # Aggregate mains = background standby + sum(target appliances) + unmetered noise
    unmetered_noise = rng.normal(0, 15, total_samples) + rng.exponential(30, total_samples)
    mains_power = base_standby_watts + unmetered_noise
    for app in appliances:
        mains_power += data[app]

    data["mains"] = np.clip(mains_power, a_min=10.0, a_max=None)

    df = pd.DataFrame(data, index=timestamps)
    df.index.name = "datetime"
    return df


def generate_benchmark_dataset(
    output_dir: str = "data/processed",
    num_houses: int = 3,
    days_per_house: int = 6,
) -> Dict[str, str]:
    """Generates synthetic multi-house datasets (House 1, 2 [held out], 3) and saves to CSV."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    saved_files = {}

    for h in range(1, num_houses + 1):
        seed = 100 + h * 37
        df = generate_synthetic_house(
            num_days=days_per_house,
            seed=seed,
            base_standby_watts=70.0 + h * 15.0,
        )
        file_path = out_path / f"redd_simulated_house_{h}.csv"
        df.to_csv(file_path)
        saved_files[f"house_{h}"] = str(file_path)
        print(f"Generated synthetic House {h} ({len(df)} samples) -> {file_path}")

    return saved_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic benchmark smart-meter data")
    parser.add_argument("--out_dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--houses", type=int, default=3, help="Number of houses to simulate")
    parser.add_argument("--days", type=int, default=7, help="Days per house")
    args = parser.parse_args()
    generate_benchmark_dataset(output_dir=args.out_dir, num_houses=args.houses, days_per_house=args.days)
