"""Data loading, alignment, 6s resampling, windowing, and active normalization pipeline.

Supports:
- Dynamic labels.dat parsing with typo matching ('dishwaser', 'washer_dryer', 'refrigerator').
- Appliance/house coverage table inspection.
- Active-period normalization avoiding zero-stretching distortion.
- Fixed-length (L=599) sliding windows with appliance presence masking.
"""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import (
    DEFAULT_APPLIANCE_THRESHOLDS,
    DEFAULT_APPLIANCES,
    REDD_CHANNEL_MAP,
    UKDALE_CHANNEL_MAP,
    NILMConfig,
)


# Canonical name synonyms mapping raw labels to standard 4 target appliances
LABEL_SYNONYMS: Dict[str, str] = {
    # Fridge
    "refrigerator": "fridge",
    "fridge": "fridge",
    "fridge_freezer": "fridge",
    "freezer": "fridge",
    # Microwave
    "microwave": "microwave",
    # Dishwasher (including the well-known official REDD typo 'dishwaser')
    "dishwasher": "dishwasher",
    "dishwaser": "dishwasher",
    "dish_washer": "dishwasher",
    # Washing machine
    "washer_dryer": "washing_machine",
    "washer": "washing_machine",
    "washing_machine": "washing_machine",
    "dryer": "washing_machine",
    # Kettle (UK-DALE)
    "kettle": "kettle",
    # Mains
    "mains": "mains",
    "main": "mains",
    "aggregate": "mains",
    "site_meter": "mains",
}


def parse_labels_dat(house_dir: Union[str, Path]) -> Dict[str, List[int]]:
    """Parses a house's labels.dat file and maps channels to target appliances.

    Handles multi-channel appliances (e.g. washer_dryer on ch 10 & 20) and typos like 'dishwaser'.
    """
    house_dir = Path(house_dir)
    labels_file = house_dir / "labels.dat"

    channel_mapping: Dict[str, List[int]] = {
        "mains": [1, 2],  # REDD default
        "fridge": [],
        "microwave": [],
        "dishwasher": [],
        "washing_machine": [],
        "kettle": [],
    }

    if not labels_file.exists():
        return channel_mapping

    with open(labels_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                chan_id = int(parts[0])
            except ValueError:
                continue

            raw_label = parts[1].lower().strip()
            standard_app = LABEL_SYNONYMS.get(raw_label)

            if standard_app in channel_mapping:
                if chan_id not in channel_mapping[standard_app]:
                    channel_mapping[standard_app].append(chan_id)

    return channel_mapping


def build_coverage_table(
    raw_dir: Union[str, Path],
    appliances: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Scans all house directories and returns a DataFrame showing appliance channel coverage."""
    raw_path = Path(raw_dir)
    appliances = appliances or DEFAULT_APPLIANCES

    rows = []
    for h_dir in sorted(raw_path.glob("house_*")):
        try:
            h_id = int(h_dir.name.split("_")[-1])
        except ValueError:
            continue

        mapping = parse_labels_dat(h_dir)
        # Fall back to REDD_CHANNEL_MAP if labels.dat had no channels mapped
        if not any(mapping[app] for app in appliances):
            mapping = REDD_CHANNEL_MAP.get(h_id, mapping)

        row = {"House": f"House {h_id}"}
        mains_ch = mapping.get("mains", [1, 2])
        row["Mains"] = f"✓ (ch {','.join(map(str, mains_ch))})" if mains_ch else "✗"

        for app in appliances:
            chans = mapping.get(app, [])
            if chans:
                row[app.replace("_", " ").title()] = f"✓ (ch {','.join(map(str, chans))})"
            else:
                row[app.replace("_", " ").title()] = "✗ Missing"

        rows.append(row)

    coverage_df = pd.DataFrame(rows)
    return coverage_df


@dataclass
class NormalizationParams:
    """Stores normalization statistics computed on training splits."""

    mains_mean: float
    mains_std: float
    appliance_stats: Dict[str, Dict[str, float]]  # {app: {"active_mean": ..., "active_std": ..., "threshold": ...}}

    def normalize_mains(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mains_mean) / (self.mains_std + 1e-8)

    def denormalize_mains(self, x_norm: np.ndarray) -> np.ndarray:
        return x_norm * (self.mains_std + 1e-8) + self.mains_mean

    def normalize_appliance(self, y: np.ndarray, appliance: str) -> np.ndarray:
        stats = self.appliance_stats.get(appliance, {})
        mean = stats.get("active_mean", 0.0)
        std = stats.get("active_std", 1.0)
        return (y - mean) / (std + 1e-8)

    def denormalize_appliance(self, y_norm: np.ndarray, appliance: str) -> np.ndarray:
        stats = self.appliance_stats.get(appliance, {})
        mean = stats.get("active_mean", 0.0)
        std = stats.get("active_std", 1.0)
        raw = y_norm * (std + 1e-8) + mean
        return np.clip(raw, a_min=0.0, a_max=None)

    def save_json(self, filepath: Union[str, Path]) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load_json(cls, filepath: Union[str, Path]) -> "NormalizationParams":
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(
            mains_mean=float(data["mains_mean"]),
            mains_std=float(data["mains_std"]),
            appliance_stats=data["appliance_stats"],
        )


def compute_normalization_params(
    train_df: pd.DataFrame,
    appliances: List[str],
    thresholds: Optional[Dict[str, float]] = None,
) -> NormalizationParams:
    """Computes global mains normalization and per-appliance active-period normalization."""
    thresholds = thresholds or DEFAULT_APPLIANCE_THRESHOLDS

    mains_series = train_df["mains"].dropna()
    mains_mean = float(mains_series.mean())
    mains_std = float(mains_series.std()) if mains_series.std() > 0 else 1.0

    app_stats: Dict[str, Dict[str, float]] = {}
    for app in appliances:
        thresh = thresholds.get(app, 20.0)
        if app in train_df.columns:
            series = train_df[app].dropna()
            # Standard NILM convention: mean and std over ACTIVE periods only (power >= threshold)
            active_vals = series[series >= thresh]
            if len(active_vals) > 10:
                act_mean = float(active_vals.mean())
                act_std = float(active_vals.std()) if active_vals.std() > 0 else 1.0
            else:
                act_mean = float(series.mean()) if len(series) > 0 else 100.0
                act_std = float(series.std()) if len(series) > 1 and series.std() > 0 else 50.0
        else:
            act_mean = 100.0
            act_std = 50.0

        app_stats[app] = {
            "active_mean": act_mean,
            "active_std": act_std,
            "threshold": float(thresh),
        }

    return NormalizationParams(
        mains_mean=mains_mean,
        mains_std=mains_std,
        appliance_stats=app_stats,
    )


def load_redd_house(
    house_dir: Union[str, Path],
    house_id: int,
    appliances: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, bool]]:
    """Parses raw REDD channel_*.dat files for a given house.

    Returns:
        df_aligned: DataFrame indexed by UTC datetime with columns ['mains'] + appliances
        presence: Dict indicating whether each appliance was present in this house
    """
    house_dir = Path(house_dir)
    appliances = appliances or DEFAULT_APPLIANCES

    # Check labels.dat dynamically
    mapping = parse_labels_dat(house_dir)
    # If dynamic mapping found no submeters, fall back to REDD_CHANNEL_MAP
    if not any(mapping[app] for app in appliances):
        mapping = REDD_CHANNEL_MAP.get(house_id, REDD_CHANNEL_MAP[1])

    channels_to_read = set(mapping.get("mains", [1, 2]))
    presence: Dict[str, bool] = {}
    for app in appliances:
        chans = mapping.get(app, [])
        if chans:
            channels_to_read.update(chans)
            presence[app] = True
        else:
            presence[app] = False

    channel_series: Dict[int, pd.Series] = {}
    for ch in channels_to_read:
        ch_file = house_dir / f"channel_{ch}.dat"
        if not ch_file.exists():
            continue
        df_ch = pd.read_csv(
            ch_file,
            sep=r"\s+",
            header=None,
            names=["timestamp", "power"],
            dtype={"timestamp": "int64", "power": "float32"},
        )
        df_ch["datetime"] = pd.to_datetime(df_ch["timestamp"], unit="s", utc=True)
        df_ch = df_ch.drop_duplicates(subset=["datetime"]).set_index("datetime")
        channel_series[ch] = df_ch["power"]

    if not channel_series:
        raise FileNotFoundError(f"No valid channel_*.dat files found in {house_dir}")

    combined = pd.DataFrame(channel_series)

    # Compute aggregate mains (sum of mains channels, typically ch1 and ch2)
    mains_cols = [c for c in mapping.get("mains", [1, 2]) if c in combined.columns]
    df_aligned = pd.DataFrame(index=combined.index)
    df_aligned["mains"] = combined[mains_cols].sum(axis=1)

    for app in appliances:
        if presence.get(app, False):
            app_cols = [c for c in mapping[app] if c in combined.columns]
            if app_cols:
                df_aligned[app] = combined[app_cols].sum(axis=1)
            else:
                df_aligned[app] = 0.0
                presence[app] = False
        else:
            df_aligned[app] = 0.0

    return df_aligned, presence


def resample_and_clean(
    df: pd.DataFrame,
    sample_period_seconds: int = 6,
    max_gap_fill_samples: int = 3,
) -> pd.DataFrame:
    """Resamples dataframe to fixed step (6 seconds) and forward-fills small gaps."""
    rule = f"{sample_period_seconds}s"
    df_resampled = df.resample(rule).mean()
    if max_gap_fill_samples > 0:
        df_resampled = df_resampled.ffill(limit=max_gap_fill_samples)
    return df_resampled


def create_sliding_windows(
    df: pd.DataFrame,
    appliances: List[str],
    norm_params: NormalizationParams,
    window_length: int = 599,
    stride: int = 149,
    appliance_presence: Optional[Dict[str, bool]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates fixed-length sliding windows for aggregate and target loads.

    Returns:
        X: (N, 599, 1) normalized aggregate windows.
        Y_power: (N, 599, num_appliances) normalized power windows.
        Y_onoff: (N, 599, num_appliances) binary on/off state labels.
        App_mask: (N, num_appliances) binary mask of appliance availability.
    """
    cols = ["mains"] + appliances
    df_subset = df[cols].copy()

    valid_mask = ~df_subset.isna().any(axis=1).values
    total_len = len(df_subset)

    mains_raw = df_subset["mains"].values.astype(np.float32)
    mains_norm = norm_params.normalize_mains(mains_raw)

    app_power_norm_list = []
    app_onoff_list = []
    app_mask_list = []

    for app in appliances:
        is_present = appliance_presence.get(app, True) if appliance_presence else True
        raw_p = df_subset[app].values.astype(np.float32)
        norm_p = norm_params.normalize_appliance(raw_p, app)
        thresh = norm_params.appliance_stats.get(app, {}).get("threshold", 20.0)
        onoff = (raw_p >= thresh).astype(np.float32)
        app_power_norm_list.append(norm_p)
        app_onoff_list.append(onoff)
        app_mask_list.append(1.0 if is_present else 0.0)

    targets_power = np.stack(app_power_norm_list, axis=-1)  # (total_len, num_apps)
    targets_onoff = np.stack(app_onoff_list, axis=-1)       # (total_len, num_apps)
    mask_vector = np.array(app_mask_list, dtype=np.float32) # (num_apps,)

    x_windows = []
    y_power_windows = []
    y_onoff_windows = []
    mask_windows = []

    for start_idx in range(0, total_len - window_length + 1, stride):
        end_idx = start_idx + window_length
        if np.all(valid_mask[start_idx:end_idx]):
            x_windows.append(mains_norm[start_idx:end_idx, np.newaxis])
            y_power_windows.append(targets_power[start_idx:end_idx])
            y_onoff_windows.append(targets_onoff[start_idx:end_idx])
            mask_windows.append(mask_vector)

    if not x_windows:
        return (
            np.empty((0, window_length, 1), dtype=np.float32),
            np.empty((0, window_length, len(appliances)), dtype=np.float32),
            np.empty((0, window_length, len(appliances)), dtype=np.float32),
            np.empty((0, len(appliances)), dtype=np.float32),
        )

    return (
        np.array(x_windows, dtype=np.float32),
        np.array(y_power_windows, dtype=np.float32),
        np.array(y_onoff_windows, dtype=np.float32),
        np.array(mask_windows, dtype=np.float32),
    )


class NILMDataset(Dataset):
    """PyTorch Dataset yielding (mains_window, power_targets, onoff_targets, app_mask)."""

    def __init__(
        self,
        x: np.ndarray,
        y_power: np.ndarray,
        y_onoff: np.ndarray,
        app_mask: Optional[np.ndarray] = None,
    ):
        self.x = torch.from_numpy(x).float()
        self.y_power = torch.from_numpy(y_power).float()
        self.y_onoff = torch.from_numpy(y_onoff).float()
        if app_mask is not None:
            self.app_mask = torch.from_numpy(app_mask).float()
        else:
            self.app_mask = torch.ones((len(x), y_power.shape[-1]), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y_power[idx], self.y_onoff[idx], self.app_mask[idx]
