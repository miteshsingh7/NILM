"""Synthetic aggregate data augmentation module based on Kelly & Knottenbelt (2015).

Methodology:
1. Extract contiguous active snippets for each appliance across all training houses.
2. Extract quiet background aggregate segments (baseload + noise, no target-appliance activity).
3. Synthesize training windows:
   - Start from a random quiet background.
   - For a designated target appliance, 50% probability of inserting an active snippet.
   - For each other appliance, 25% probability of inserting a distractor snippet.
   - Sum inserted appliances into the aggregate mains.
   - Exact ground truth is tracked for regression and binary on/off labels.
4. Normalize and combine with real training windows.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.data_pipeline import NILMDataset, NormalizationParams


def extract_activation_snippets(
    train_dfs: List[Tuple[pd.DataFrame, Dict[str, bool]]],
    appliances: List[str],
    thresholds: Dict[str, float],
    padding: int = 8,
    max_length: int = 599,
    min_length: int = 3,
) -> Dict[str, List[np.ndarray]]:
    """Extracts contiguous active appliance snippets pooled across all training houses.

    Args:
        train_dfs: List of (df, presence_dict) tuples for training houses.
        appliances: List of target appliance names.
        thresholds: Dictionary of appliance on/off power thresholds.
        padding: Samples to include before/after the active period to capture transition edges.
        max_length: Maximum allowed snippet length (clamped or sub-sliced to fit in window).
        min_length: Minimum contiguous active samples to qualify as a valid event.

    Returns:
        snippets_pool: Dict mapping appliance name to list of 1D numpy power arrays.
    """
    snippets_pool: Dict[str, List[np.ndarray]] = {app: [] for app in appliances}

    for df, presence in train_dfs:
        for app in appliances:
            if not presence.get(app, True) or app not in df.columns:
                continue

            series = df[app].values.astype(np.float32)
            thresh = float(thresholds.get(app, 20.0))
            is_active = (series >= thresh).astype(np.int32)

            # Find contiguous runs of is_active == 1
            diff = np.diff(np.pad(is_active, (1, 1), mode="constant", constant_values=0))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]

            for s_idx, e_idx in zip(starts, ends):
                active_len = e_idx - s_idx
                if active_len < min_length:
                    continue

                # Add padding before and after
                pad_s = max(0, s_idx - padding)
                pad_e = min(len(series), e_idx + padding)
                snippet = series[pad_s:pad_e].copy()

                # Discard if snippet contains NaNs
                if np.isnan(snippet).any():
                    continue

                # Clean any negative noise
                snippet = np.maximum(snippet, 0.0)

                # If snippet exceeds max_length, split into sub-windows or clamp
                if len(snippet) > max_length:
                    for sub_start in range(0, len(snippet) - max_length + 1, max_length // 2):
                        sub_snippet = snippet[sub_start : sub_start + max_length]
                        if not np.isnan(sub_snippet).any():
                            snippets_pool[app].append(sub_snippet)
                else:
                    snippets_pool[app].append(snippet)

    return snippets_pool


def extract_quiet_baseloads(
    train_dfs: List[Tuple[pd.DataFrame, Dict[str, bool]]],
    appliances: List[str],
    thresholds: Dict[str, float],
    window_length: int = 599,
    stride: int = 200,
    max_candidates: int = 5000,
) -> List[np.ndarray]:
    """Extracts quiet background segments from training mains where all target appliances are off.

    Args:
        train_dfs: Training house splits.
        appliances: List of target appliances.
        thresholds: Dictionary of on/off thresholds.
        window_length: Length of window (default 599).
        stride: Stride between candidate quiet windows.
        max_candidates: Max quiet windows to retain.

    Returns:
        quiet_windows: List of 1D numpy arrays of raw quiet mains power.
    """
    quiet_windows: List[np.ndarray] = []

    for df, presence in train_dfs:
        if "mains" not in df.columns:
            continue

        # Build mask of timesteps where all present target appliances are < threshold
        quiet_mask = np.ones(len(df), dtype=bool)
        for app in appliances:
            if presence.get(app, True) and app in df.columns:
                thresh = float(thresholds.get(app, 20.0))
                quiet_mask &= (df[app].values < thresh)

        mains = df["mains"].values.astype(np.float32)
        total_len = len(df)

        for s_idx in range(0, total_len - window_length + 1, stride):
            e_idx = s_idx + window_length
            if np.all(quiet_mask[s_idx:e_idx]) and not np.isnan(mains[s_idx:e_idx]).any():
                quiet_windows.append(mains[s_idx:e_idx].copy())
                if len(quiet_windows) >= max_candidates:
                    return quiet_windows

    # Fallback: if very few strictly quiet windows exist, find windows with lowest target power sum
    if len(quiet_windows) < 10:
        for df, _ in train_dfs:
            if "mains" not in df.columns:
                continue
            mains = df["mains"].values.astype(np.float32)
            for s_idx in range(0, len(mains) - window_length + 1, stride * 2):
                seg = mains[s_idx : s_idx + window_length]
                if not np.isnan(seg).any() and np.mean(seg) < 400.0:
                    quiet_windows.append(seg.copy())

    return quiet_windows


def generate_synthetic_windows(
    snippets_pool: Dict[str, List[np.ndarray]],
    quiet_baseloads: List[np.ndarray],
    appliances: List[str],
    thresholds: Dict[str, float],
    norm_params: NormalizationParams,
    num_windows: int,
    window_length: int = 599,
    target_prob: float = 0.50,
    distractor_prob: float = 0.25,
    exclude_appliances: Optional[List[str]] = None,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthesizes windows according to Kelly & Knottenbelt (2015).

    Appliances in exclude_appliances (e.g. 'fridge') are never injected, and their
    appliance mask in synthetic windows is set to 0.0 so the model trains those heads
    on real data only.
    """
    rng = np.random.RandomState(random_seed)
    num_apps = len(appliances)
    excluded = set(exclude_appliances or [])

    # Target candidate pool excludes non-burst / continuous loads (e.g. fridge)
    target_candidates = [app for app in appliances if app not in excluded]
    if not target_candidates:
        target_candidates = list(appliances)

    if not quiet_baseloads:
        quiet_baseloads = [np.full(window_length, 150.0, dtype=np.float32) + rng.normal(0, 10, window_length).astype(np.float32)]

    x_list: List[np.ndarray] = []
    yp_list: List[np.ndarray] = []
    yo_list: List[np.ndarray] = []
    mask_list: List[np.ndarray] = []

    # Mask vector: 0.0 for excluded appliances (no loss on synthetic data), 1.0 for augmented appliances
    mask_vector = np.array([0.0 if app in excluded else 1.0 for app in appliances], dtype=np.float32)

    for i in range(num_windows):
        # 1. Sample a random quiet baseload window
        b_idx = rng.randint(0, len(quiet_baseloads))
        mains_raw = quiet_baseloads[b_idx].copy()
        if len(mains_raw) != window_length:
            mains_raw = np.resize(mains_raw, window_length)

        # 2. Designated target appliance from the candidate pool
        target_app = target_candidates[i % len(target_candidates)]

        # 3. Create raw power targets
        app_raw_power = np.zeros((window_length, num_apps), dtype=np.float32)

        for k, app_name in enumerate(appliances):
            if app_name in excluded:
                continue

            prob = target_prob if app_name == target_app else distractor_prob
            snippets = snippets_pool.get(app_name, [])

            if rng.uniform() < prob and len(snippets) > 0:
                s_idx = rng.randint(0, len(snippets))
                snippet = snippets[s_idx]
                s_len = min(len(snippet), window_length)
                s_data = snippet[:s_len]

                # Random placement in window
                t_start = rng.randint(0, window_length - s_len + 1)
                t_end = t_start + s_len

                # Add snippet to aggregate mains and target power
                mains_raw[t_start:t_end] += s_data
                app_raw_power[t_start:t_end, k] += s_data

        # 4. Generate on/off targets
        app_onoff = np.zeros((window_length, num_apps), dtype=np.float32)
        app_norm_power = np.zeros((window_length, num_apps), dtype=np.float32)

        for k, app_name in enumerate(appliances):
            thresh = float(thresholds.get(app_name, 20.0))
            app_onoff[:, k] = (app_raw_power[:, k] >= thresh).astype(np.float32)
            app_norm_power[:, k] = norm_params.normalize_appliance(app_raw_power[:, k], app_name)

        # 5. Normalize mains
        mains_norm = norm_params.normalize_mains(mains_raw).reshape(window_length, 1)

        x_list.append(mains_norm)
        yp_list.append(app_norm_power)
        yo_list.append(app_onoff)
        mask_list.append(mask_vector)

    X_synth = np.nan_to_num(np.array(x_list, dtype=np.float32), nan=0.0)
    Yp_synth = np.nan_to_num(np.array(yp_list, dtype=np.float32), nan=0.0)
    Yo_synth = np.nan_to_num(np.array(yo_list, dtype=np.float32), nan=0.0)
    Mask_synth = np.nan_to_num(np.array(mask_list, dtype=np.float32), nan=1.0)

    assert not np.isnan(X_synth).any(), "X_synth contains NaNs!"
    assert not np.isnan(Yp_synth).any(), "Yp_synth contains NaNs!"
    assert not np.isnan(Yo_synth).any(), "Yo_synth contains NaNs!"

    return X_synth, Yp_synth, Yo_synth, Mask_synth


def build_augmented_training_dataset(
    real_train_dataset: NILMDataset,
    train_dfs: List[Tuple[pd.DataFrame, Dict[str, bool]]],
    appliances: List[str],
    thresholds: Dict[str, float],
    norm_params: NormalizationParams,
    synthetic_ratio: float = 1.0,
    window_length: int = 599,
    exclude_appliances: Optional[List[str]] = None,
    random_seed: int = 42,
) -> Tuple[NILMDataset, Dict[str, any]]:
    """Builds a mixed training dataset combining real oversampled windows and Kelly & Knottenbelt synthetic windows.

    Args:
        real_train_dataset: The base real NILMDataset.
        train_dfs: Training house dataframes for snippet & quiet period extraction.
        appliances: List of target appliances.
        thresholds: On/off thresholds dictionary.
        norm_params: Fitted normalization parameters.
        synthetic_ratio: Ratio of synthetic windows relative to real windows (1.0 = 50% real / 50% synthetic).
        window_length: Window length (599).
        random_seed: Random seed.

    Returns:
        combined_dataset: NILMDataset containing real + synthetic windows.
        audit_info: Dictionary with snippet counts and window statistics.
    """
    num_real = len(real_train_dataset)
    num_synth = int(num_real * synthetic_ratio)

    print(f"\n[Synthetic Augmentation] Extracting active snippets across training houses...")
    snippets_pool = extract_activation_snippets(
        train_dfs=train_dfs,
        appliances=appliances,
        thresholds=thresholds,
        max_length=window_length,
    )
    for app in appliances:
        print(f"  -> {app:<15s}: {len(snippets_pool[app]):,} activation snippets extracted")

    print(f"[Synthetic Augmentation] Extracting quiet baseload segments...")
    quiet_baseloads = extract_quiet_baseloads(
        train_dfs=train_dfs,
        appliances=appliances,
        thresholds=thresholds,
        window_length=window_length,
    )
    print(f"  -> Quiet baseload library: {len(quiet_baseloads):,} segments")

    print(f"[Synthetic Augmentation] Generating {num_synth:,} synthetic training windows (Kelly & Knottenbelt)...")
    X_s, Yp_s, Yo_s, M_s = generate_synthetic_windows(
        snippets_pool=snippets_pool,
        quiet_baseloads=quiet_baseloads,
        appliances=appliances,
        thresholds=thresholds,
        norm_params=norm_params,
        num_windows=num_synth,
        window_length=window_length,
        exclude_appliances=exclude_appliances,
        random_seed=random_seed,
    )

    # Real data arrays from dataset
    X_r = real_train_dataset.x.numpy()
    Yp_r = real_train_dataset.y_power.numpy()
    Yo_r = real_train_dataset.y_onoff.numpy()
    M_r = real_train_dataset.app_mask.numpy()

    # Concatenate real + synthetic
    X_comb = np.concatenate([X_r, X_s], axis=0)
    Yp_comb = np.concatenate([Yp_r, Yp_s], axis=0)
    Yo_comb = np.concatenate([Yo_r, Yo_s], axis=0)
    M_comb = np.concatenate([M_r, M_s], axis=0)

    combined_dataset = NILMDataset(
        x=X_comb,
        y_power=Yp_comb,
        y_onoff=Yo_comb,
        app_mask=M_comb,
    )

    audit_info = {
        "num_real_windows": num_real,
        "num_synthetic_windows": num_synth,
        "total_combined_windows": len(combined_dataset),
        "snippets_per_appliance": {app: len(snippets_pool[app]) for app in appliances},
        "quiet_baseload_count": len(quiet_baseloads),
    }

    print(f"[Synthetic Augmentation] Complete: {num_real:,} real + {num_synth:,} synthetic = {len(combined_dataset):,} total windows.\n")
    return combined_dataset, audit_info
