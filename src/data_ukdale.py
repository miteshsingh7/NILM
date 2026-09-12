"""Data loading, alignment, and dataset preparation for UK-DALE disaggregated dataset.

Supports:
- Dynamic labels.dat parsing with UK-DALE naming conventions (fridge_freezer, dish_washer, washer_dryer).
- Clean channel alignment to 6-second sampling without interpolation artifacts.
- Caching processed per-house CSVs into data/processed/ukdale_real_house_{h}.csv.
- Active normalization parameter computation and sliding window dataset generation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import (
    DEFAULT_APPLIANCE_THRESHOLDS,
    DEFAULT_APPLIANCES,
    UKDALE_CHANNEL_MAP,
    NILMConfig,
)
from src.data_pipeline import (
    LABEL_SYNONYMS,
    NILMDataset,
    NormalizationParams,
    compute_normalization_params,
    create_sliding_windows,
    parse_labels_dat,
    resample_and_clean,
)


def load_ukdale_house(
    house_dir: Union[str, Path],
    house_id: int,
    appliances: Optional[List[str]] = None,
    sample_period_seconds: int = 6,
    max_gap_fill_samples: int = 3,
) -> Tuple[pd.DataFrame, Dict[str, bool]]:
    """Loads, resamples to 6s, and aligns raw UK-DALE channel_*.dat files for a given house.

    Returns:
        df_aligned: DataFrame indexed by UTC datetime with columns ['mains'] + appliances
        presence: Dict indicating whether each appliance was present and cleanly metered in this house
    """
    house_dir = Path(house_dir)
    appliances = appliances or DEFAULT_APPLIANCES

    # 1. Parse labels.dat
    mapping = parse_labels_dat(house_dir)

    # For House 4, channel 5 is 'freezer' and channel 6 is 'washing_machine_microwave_breadmaker'
    # These are NOT clean standalone targets, so we enforce UKDALE_CHANNEL_MAP for House 4
    if house_id == 4:
        mapping = UKDALE_CHANNEL_MAP[4]
    elif not any(mapping.get(app, []) for app in appliances):
        mapping = UKDALE_CHANNEL_MAP.get(house_id, {"mains": [1]})

    mains_channels = mapping.get("mains", [1])
    if not mains_channels:
        mains_channels = [1]

    channels_to_read = set(mains_channels)
    presence: Dict[str, bool] = {}
    for app in appliances:
        chans = mapping.get(app, [])
        if chans:
            channels_to_read.update(chans)
            presence[app] = True
        else:
            presence[app] = False

    rule = f"{sample_period_seconds}s"
    channel_series: Dict[int, pd.Series] = {}
    for ch in sorted(channels_to_read):
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
        s = df_ch.drop_duplicates(subset=["datetime"]).set_index("datetime")["power"]
        # Resample each channel individually to 6-second bins before joining
        s_resampled = s.resample(rule).mean()
        if max_gap_fill_samples > 0:
            s_resampled = s_resampled.ffill(limit=max_gap_fill_samples)
        channel_series[ch] = s_resampled

    if not channel_series or not any(ch in channel_series for ch in mains_channels):
        raise FileNotFoundError(f"No valid mains channel_*.dat found in {house_dir}")

    # Align across channels (all are already on the exact same 6s grid)
    combined = pd.DataFrame(channel_series)

    # Compute aggregate mains
    mains_cols = [c for c in mains_channels if c in combined.columns]
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

    # Drop trailing/leading NaNs in mains
    df_aligned = df_aligned.dropna(subset=["mains"])
    return df_aligned, presence


def get_ukdale_presence(
    house_dir: Union[str, Path],
    house_id: int,
    appliances: Optional[List[str]] = None,
) -> Dict[str, bool]:
    """Determines presence of target appliances in a UK-DALE house directly from metadata."""
    house_dir = Path(house_dir)
    appliances = appliances or DEFAULT_APPLIANCES
    mapping = parse_labels_dat(house_dir)
    if house_id == 4:
        mapping = UKDALE_CHANNEL_MAP[4]
    elif not any(mapping.get(app, []) for app in appliances):
        mapping = UKDALE_CHANNEL_MAP.get(house_id, {"mains": [1]})
    return {app: bool(mapping.get(app, [])) for app in appliances}


def build_ukdale_coverage_table(
    raw_dir: Union[str, Path],
    appliances: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Scans all UK-DALE house directories and returns a DataFrame showing appliance channel coverage."""
    raw_path = Path(raw_dir)
    appliances = appliances or DEFAULT_APPLIANCES

    rows = []
    for h_dir in sorted(raw_path.glob("house_*")):
        try:
            h_id = int(h_dir.name.split("_")[-1])
        except ValueError:
            continue

        mapping = parse_labels_dat(h_dir)
        if h_id == 4:
            mapping = UKDALE_CHANNEL_MAP[4]
        elif not any(mapping.get(app, []) for app in appliances):
            mapping = UKDALE_CHANNEL_MAP.get(h_id, {"mains": [1]})

        row = {"House": f"House {h_id}"}
        mains_ch = mapping.get("mains", [1])
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


def print_ukdale_active_sample_audit(
    house_dfs: Dict[int, pd.DataFrame],
    cfg: NILMConfig,
    fold: int,
) -> pd.DataFrame:
    """Prints active-sample counts per appliance for training pool vs held-out house in UK-DALE."""
    appliances = cfg.appliances
    all_houses = sorted(house_dfs.keys())
    ho_df = house_dfs[fold]
    tr_houses = [h for h in all_houses if h != fold]
    tr_df = pd.concat([house_dfs[h] for h in tr_houses], axis=0)

    print(f"\n================ FOLD {fold} ACTIVE SAMPLE AUDIT (UK-DALE) ================")
    print(f"Held-Out Generalization House: House {fold} ({len(ho_df):,} samples)")
    print(f"Training Pool Houses: {', '.join(f'House {h}' for h in tr_houses)} ({len(tr_df):,} samples)")
    print("----------------------------------------------------------------------------------------")

    rows = []
    for app in appliances:
        thresh = cfg.get_threshold(app)
        tr_cnt = int((tr_df[app] >= thresh).sum()) if app in tr_df.columns else 0
        tr_pct = tr_cnt / len(tr_df) * 100 if len(tr_df) > 0 else 0.0
        ho_cnt = int((ho_df[app] >= thresh).sum()) if app in ho_df.columns else 0
        ho_pct = ho_cnt / len(ho_df) * 100 if len(ho_df) > 0 else 0.0

        rows.append({
            "Appliance": app,
            "Training Pool Active Samples": f"{tr_cnt:,} / {len(tr_df):,} ({tr_pct:.2f}%)",
            "Held-Out House Active Samples": f"{ho_cnt:,} / {len(ho_df):,} ({ho_pct:.2f}%)",
        })

    audit_df = pd.DataFrame(rows)
    print(audit_df.to_string(index=False))
    print("========================================================================================\n")
    return audit_df


def prepare_ukdale_datasets(
    config: NILMConfig,
    data_dir: str = "data/processed",
    ukdale_dir: str = "data/raw/ukdale",
) -> Tuple[NILMDataset, NILMDataset, NILMDataset, NormalizationParams]:
    """Loads, resamples, caches, and windows UK-DALE houses for LOHO-CV training and evaluation."""
    appliances = config.appliances
    ukdale_path = Path(ukdale_dir)

    if not ukdale_path.exists():
        raise FileNotFoundError(f"UK-DALE directory not found: {ukdale_dir}")

    # Build coverage table
    coverage_df = build_ukdale_coverage_table(ukdale_path, appliances=appliances)
    print("\n================ UK-DALE APPLIANCE / HOUSE COVERAGE TABLE ================")
    print(coverage_df.to_string(index=False))
    print("==========================================================================\n")

    house_dfs: Dict[int, pd.DataFrame] = {}
    house_presences: Dict[int, Dict[str, bool]] = {}

    for h_dir in sorted(ukdale_path.glob("house_*")):
        try:
            h_id = int(h_dir.name.split("_")[-1])
            cached_parquet = Path(data_dir) / f"ukdale_real_house_{h_id}.parquet"
            cached_csv = Path(data_dir) / f"ukdale_real_house_{h_id}.csv"
            if cached_parquet.exists():
                df_clean = pd.read_parquet(cached_parquet)
                presence = get_ukdale_presence(h_dir, h_id, appliances=appliances)
                print(f"Loaded cached UK-DALE House {h_id}: {len(df_clean):,} samples @ 6s ({len(df_clean)*6/3600/24:.1f} days)")
            elif cached_csv.exists():
                df_clean = pd.read_csv(cached_csv, index_col=0, parse_dates=True)
                presence = get_ukdale_presence(h_dir, h_id, appliances=appliances)
                print(f"Loaded cached UK-DALE House {h_id}: {len(df_clean):,} samples @ 6s ({len(df_clean)*6/3600/24:.1f} days)")
            else:
                df_clean, presence = load_ukdale_house(
                    h_dir,
                    h_id,
                    appliances=appliances,
                    sample_period_seconds=config.sample_period_seconds,
                    max_gap_fill_samples=config.max_gap_fill_samples,
                )
                df_clean.to_parquet(cached_parquet)
                print(f"Saved & loaded UK-DALE House {h_id}: {len(df_clean):,} samples @ 6s ({len(df_clean)*6/3600/24:.1f} days)")

            house_dfs[h_id] = df_clean
            house_presences[h_id] = presence
        except Exception as e:
            print(f"Skipping {h_dir.name}: {e}")

    if not house_dfs:
        raise RuntimeError("No UK-DALE datasets could be loaded.")

    held_out_id = config.held_out_house
    train_dfs: List[Tuple[pd.DataFrame, Dict[str, bool]]] = []
    val_dfs: List[Tuple[pd.DataFrame, Dict[str, bool]]] = []
    test_held_out_dfs: List[Tuple[pd.DataFrame, Dict[str, bool]]] = []

    for h_id, df in house_dfs.items():
        presence = house_presences.get(h_id, {app: True for app in appliances})
        if h_id == held_out_id:
            test_held_out_dfs.append((df, presence))
            print(f"Held-out Generalization House: House {h_id} ({len(df):,} samples)")
        else:
            split_idx = int(len(df) * config.train_val_split_ratio)
            train_dfs.append((df.iloc[:split_idx], presence))
            val_dfs.append((df.iloc[split_idx:], presence))

    # Compute normalization statistics across training houses
    combined_train_df = pd.concat([item[0] for item in train_dfs], axis=0)
    norm_params = compute_normalization_params(
        combined_train_df,
        appliances=appliances,
        thresholds=config.thresholds,
    )

    print("\n================ NORMALIZATION PARAMETERS GENERATION ================")
    print(f"[Verification] Freshly generated norm_params from {len(combined_train_df):,} training samples.")
    print(f"[Verification] Aggregate Mains: mean = {norm_params.mains_mean:.2f} W, std = {norm_params.mains_std:.2f} W")
    for app in appliances:
        stats = norm_params.appliance_stats[app]
        print(f"  - {app:<15}: active_mean = {stats['active_mean']:.2f} W, active_std = {stats['active_std']:.2f} W, threshold = {stats['threshold']:.1f} W")
    print("=====================================================================\n")

    # Window training set with overlapping stride (L // 4)
    x_train_list, yp_train_list, yo_train_list, m_train_list = [], [], [], []
    for t_df, presence in train_dfs:
        x, yp, yo, m = create_sliding_windows(
            t_df,
            appliances=appliances,
            norm_params=norm_params,
            window_length=config.window_length,
            stride=config.train_stride,
            appliance_presence=presence,
        )
        if len(x) > 0:
            x_train_list.append(x)
            yp_train_list.append(yp)
            yo_train_list.append(yo)
            m_train_list.append(m)

    # Window validation set with non-overlapping stride (L)
    x_val_list, yp_val_list, yo_val_list, m_val_list = [], [], [], []
    for v_df, presence in val_dfs:
        x, yp, yo, m = create_sliding_windows(
            v_df,
            appliances=appliances,
            norm_params=norm_params,
            window_length=config.window_length,
            stride=config.val_test_stride,
            appliance_presence=presence,
        )
        if len(x) > 0:
            x_val_list.append(x)
            yp_val_list.append(yp)
            yo_val_list.append(yo)
            m_val_list.append(m)

    # Window held-out test set with non-overlapping stride (L)
    x_test_list, yp_test_list, yo_test_list, m_test_list = [], [], [], []
    for ho_df, presence in test_held_out_dfs:
        x, yp, yo, m = create_sliding_windows(
            ho_df,
            appliances=appliances,
            norm_params=norm_params,
            window_length=config.window_length,
            stride=config.val_test_stride,
            appliance_presence=presence,
        )
        if len(x) > 0:
            x_test_list.append(x)
            yp_test_list.append(yp)
            yo_test_list.append(yo)
            m_test_list.append(m)

    train_ds = NILMDataset(
        np.concatenate(x_train_list, axis=0) if x_train_list else np.empty((0, config.window_length, 1), dtype=np.float32),
        np.concatenate(yp_train_list, axis=0) if yp_train_list else np.empty((0, config.window_length, len(appliances)), dtype=np.float32),
        np.concatenate(yo_train_list, axis=0) if yo_train_list else np.empty((0, config.window_length, len(appliances)), dtype=np.float32),
        np.concatenate(m_train_list, axis=0) if m_train_list else np.empty((0, len(appliances)), dtype=np.float32),
    )
    val_ds = NILMDataset(
        np.concatenate(x_val_list, axis=0) if x_val_list else np.empty((0, config.window_length, 1), dtype=np.float32),
        np.concatenate(yp_val_list, axis=0) if yp_val_list else np.empty((0, config.window_length, len(appliances)), dtype=np.float32),
        np.concatenate(yo_val_list, axis=0) if yo_val_list else np.empty((0, config.window_length, len(appliances)), dtype=np.float32),
        np.concatenate(m_val_list, axis=0) if m_val_list else np.empty((0, len(appliances)), dtype=np.float32),
    )
    test_ds = NILMDataset(
        np.concatenate(x_test_list, axis=0) if x_test_list else np.empty((0, config.window_length, 1), dtype=np.float32),
        np.concatenate(yp_test_list, axis=0) if yp_test_list else np.empty((0, config.window_length, len(appliances)), dtype=np.float32),
        np.concatenate(yo_test_list, axis=0) if yo_test_list else np.empty((0, config.window_length, len(appliances)), dtype=np.float32),
        np.concatenate(m_test_list, axis=0) if m_test_list else np.empty((0, len(appliances)), dtype=np.float32),
    )

    print(f"Generated windows -> Train: {len(train_ds):,}, Val: {len(val_ds):,}, Held-out Test: {len(test_ds):,}")
    return train_ds, val_ds, test_ds, norm_params


def prepare_ukdale_pretraining_datasets(
    config: NILMConfig,
    data_dir: str = "data/processed",
    ukdale_dir: str = "data/raw/ukdale",
) -> Tuple[NILMDataset, NILMDataset, NormalizationParams, Dict[int, pd.DataFrame]]:
    """Loads, cleans, and windows all 5 UK-DALE houses for pretraining (no held-out house).

    80% of each house is used for training and 20% for in-distribution validation.
    Returns:
        train_ds: NILMDataset across all 5 houses
        val_ds: NILMDataset across all 5 houses
        norm_params: NormalizationParams computed exclusively on UK-DALE training data
        house_dfs: Dict of full house DataFrames for active sample auditing
    """
    appliances = config.appliances
    ukdale_path = Path(ukdale_dir)

    # Coverage table
    coverage_df = build_ukdale_coverage_table(ukdale_path, appliances=appliances)
    print("\n================ UK-DALE APPLIANCE / HOUSE COVERAGE TABLE ================")
    print(coverage_df.to_string(index=False))
    print("==========================================================================\n")

    house_dfs: Dict[int, pd.DataFrame] = {}
    house_presences: Dict[int, Dict[str, bool]] = {}

    for h_id in range(1, 6):
        h_dir = ukdale_path / f"house_{h_id}"
        cached_parquet = Path(data_dir) / f"ukdale_real_house_{h_id}.parquet"
        cached_csv = Path(data_dir) / f"ukdale_real_house_{h_id}.csv"
        if cached_parquet.exists():
            df_clean = pd.read_parquet(cached_parquet)
        elif cached_csv.exists():
            df_clean = pd.read_csv(cached_csv, index_col=0, parse_dates=True)
        else:
            df_clean, _ = load_ukdale_house(h_dir, h_id, appliances=appliances)
        presence = get_ukdale_presence(h_dir, h_id, appliances=appliances)
        house_dfs[h_id] = df_clean
        house_presences[h_id] = presence
        print(f"Loaded UK-DALE House {h_id}: {len(df_clean):,} samples @ 6s ({len(df_clean)*6/86400:.1f} days)")

    # 80/20 train/val split per house (no held-out house for pretraining)
    train_dfs: List[Tuple[pd.DataFrame, Dict[str, bool], int]] = []
    val_dfs: List[Tuple[pd.DataFrame, Dict[str, bool], int]] = []

    for h_id, df in house_dfs.items():
        presence = house_presences[h_id]
        split_idx = int(len(df) * config.train_val_split_ratio)
        train_dfs.append((df.iloc[:split_idx], presence, h_id))
        val_dfs.append((df.iloc[split_idx:], presence, h_id))

    # Compute normalization statistics across all 5 training houses
    combined_train_df = pd.concat([item[0] for item in train_dfs], axis=0)
    norm_params = compute_normalization_params(
        combined_train_df,
        appliances=appliances,
        thresholds=config.thresholds,
    )

    print("\n================ UK-DALE PRETRAINING NORMALIZATION PARAMETERS ================")
    print(f"[Verification] Freshly generated norm_params from {len(combined_train_df):,} UK-DALE training samples.")
    print(f"[Verification] Aggregate Mains: mean = {norm_params.mains_mean:.2f} W, std = {norm_params.mains_std:.2f} W")
    for app in appliances:
        stats = norm_params.appliance_stats[app]
        print(f"  - {app:<15}: active_mean = {stats['active_mean']:.2f} W, active_std = {stats['active_std']:.2f} W, threshold = {stats['threshold']:.1f} W")
    print("==============================================================================\n")

    # Sliding windows:
    # Use non-overlapping 599 stride on House 1 (which has 18.7M train samples -> 31k 1-hr windows)
    # and 299 stride on House 2 & 5 (dense sampling on smaller metered houses)
    # and 599 stride on House 3 & 4 (mains only)
    x_train_list, yp_train_list, yo_train_list, m_train_list = [], [], [], []
    for t_df, presence, h_id in train_dfs:
        s = 599 if h_id in [1, 3, 4] else 299
        x, yp, yo, m = create_sliding_windows(
            t_df,
            appliances=appliances,
            norm_params=norm_params,
            window_length=config.window_length,
            stride=s,
            appliance_presence=presence,
        )
        if len(x) > 0:
            x_train_list.append(x)
            yp_train_list.append(yp)
            yo_train_list.append(yo)
            m_train_list.append(m)

    # Non-overlapping 599 stride for validation across all houses
    x_val_list, yp_val_list, yo_val_list, m_val_list = [], [], [], []
    for v_df, presence, h_id in val_dfs:
        x, yp, yo, m = create_sliding_windows(
            v_df,
            appliances=appliances,
            norm_params=norm_params,
            window_length=config.window_length,
            stride=config.val_test_stride,
            appliance_presence=presence,
        )
        if len(x) > 0:
            x_val_list.append(x)
            yp_val_list.append(yp)
            yo_val_list.append(yo)
            m_val_list.append(m)

    train_ds = NILMDataset(
        np.concatenate(x_train_list, axis=0),
        np.concatenate(yp_train_list, axis=0),
        np.concatenate(yo_train_list, axis=0),
        np.concatenate(m_train_list, axis=0),
    )
    val_ds = NILMDataset(
        np.concatenate(x_val_list, axis=0),
        np.concatenate(yp_val_list, axis=0),
        np.concatenate(yo_val_list, axis=0),
        np.concatenate(m_val_list, axis=0),
    )

    print(f"Generated UK-DALE Pretraining Windows -> Train: {len(train_ds):,}, Val: {len(val_ds):,}")
    return train_ds, val_ds, norm_params, house_dfs

