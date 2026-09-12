"""Training script for Multi-Appliance NILM with active loss weighting and per-appliance curve tracking."""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.config import NILMConfig
from src.data_pipeline import (
    NormalizationParams,
    build_coverage_table,
    compute_normalization_params,
    create_sliding_windows,
    load_redd_house,
    resample_and_clean,
    NILMDataset,
)
from src.loss import MultiApplianceLoss
from src.model import MultiApplianceNILM
from src.sample_data import generate_benchmark_dataset
from src.utils import save_checkpoint


def seed_worker(worker_id: int) -> None:
    """Seeds DataLoader worker RNG from torch.initial_seed()."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def export_best_checkpoint(ckpt_dir: Path, epoch: int, val_loss: float, run_hash: str) -> None:
    """Immediately packages and attempts to push best checkpoint to Kaggle Dataset."""
    try:
        # Write metadata.json for Kaggle Dataset packaging
        meta_path = ckpt_dir / "dataset-metadata.json"
        meta_content = {
            "title": "NILM Transfer Checkpoint Fold 2",
            "id": "nilm-transfer-fold-2",
            "licenses": [{"name": "CC0-1.0"}]
        }
        with open(meta_path, "w") as f:
            json.dump(meta_content, f, indent=2)

        # Immediate zip archive in /kaggle/working
        working_dir = Path("/kaggle/working")
        if working_dir.exists():
            zip_dest = working_dir / "redd_transfer_fold_2_best_export"
            shutil.make_archive(str(zip_dest), "zip", str(ckpt_dir))
            # Direct copy of best_model.pt and norm_params to root of working dir
            shutil.copy(str(ckpt_dir / "best_model.pt"), str(working_dir / "best_model.pt"))
            if (ckpt_dir / "norm_params.json").exists():
                shutil.copy(str(ckpt_dir / "norm_params.json"), str(working_dir / "norm_params.json"))

        # If kaggle credentials exist, push version immediately
        kaggle_json = Path.home() / ".kaggle/kaggle.json"
        if kaggle_json.exists() or "KAGGLE_USERNAME" in os.environ:
            cmd = [
                "kaggle", "datasets", "version",
                "-p", str(ckpt_dir),
                "-m", f"Best Model Epoch {epoch} Val {val_loss:.4f} Hash {run_hash[:8]}",
                "-d", "--dir-mode", "zip"
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if res.returncode == 0:
                print(f"  --> [Dataset Sync] Automatically pushed best checkpoint to Kaggle Dataset (Epoch {epoch})")
            else:
                print(f"  --> [Dataset Sync] Kaggle CLI note: {res.stderr.strip()[:80]}")
    except Exception as e:
        # Non-blocking export
        pass


def prepare_datasets(
    config: NILMConfig,
    data_dir: str = "data/processed",
    redd_dir: Optional[str] = "data/raw/redd",
    return_raw_splits: bool = False,
) -> Union[
    Tuple[NILMDataset, NILMDataset, NILMDataset, NormalizationParams],
    Tuple[NILMDataset, NILMDataset, NILMDataset, NormalizationParams, List[Tuple[pd.DataFrame, Dict[str, bool]]], Dict[int, pd.DataFrame]],
]:
    """Loads, splits, normalizes, and windows train, val, and held-out test datasets."""
    appliances = config.appliances
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    house_dfs: Dict[int, pd.DataFrame] = {}
    house_presences: Dict[int, Dict[str, bool]] = {}

    # Check for pre-processed REDD CSVs in data_dir
    cached_files = sorted(list(Path(data_dir).glob("redd_real_house_*.csv")))
    if cached_files and (not redd_dir or not Path(redd_dir).exists() or not list(Path(redd_dir).glob("house_*"))):
        print(f"\nFound {len(cached_files)} cached processed REDD CSVs in {data_dir}. Loading directly...")
        from src.config import REDD_CHANNEL_MAP
        for f in cached_files:
            try:
                h_id = int(f.stem.split("_")[-1])
                df_clean = pd.read_csv(f, index_col=0, parse_dates=True)
                ch_map = REDD_CHANNEL_MAP.get(h_id, {})
                presence = {app: (len(ch_map.get(app, [])) > 0) for app in appliances}
                house_dfs[h_id] = df_clean
                house_presences[h_id] = presence
                print(f"Loaded cached REDD House {h_id}: {len(df_clean)} samples @ 6s ({len(df_clean)*6/3600/24:.1f} days)")
            except Exception as e:
                print(f"Error loading {f.name}: {e}")
    # Strict check for real REDD data: NO silent synthetic fallback when redd_dir is passed
    elif redd_dir:
        redd_path = Path(redd_dir)
        if not redd_path.exists() or not list(redd_path.glob("house_*")):
            raise RuntimeError(
                f"FATAL: Real REDD data directory '{redd_dir}' not found or has no house directories.\n"
                f"Run 'python -m src.download_redd' to download and unpack real REDD data.\n"
                f"Refusing to silently fall back to synthetic data."
            )

        print(f"\nFound verified real REDD directory: {redd_dir}")

        # Display coverage table
        coverage_df = build_coverage_table(redd_path, appliances=appliances)
        print("\n================ REDD APPLIANCE / HOUSE COVERAGE TABLE ================")
        print(coverage_df.to_string(index=False))
        print("=======================================================================\n")

        for h_dir in sorted(redd_path.glob("house_*")):
            try:
                h_id = int(h_dir.name.split("_")[-1])
                cached_csv = Path(data_dir) / f"redd_real_house_{h_id}.csv"
                if cached_csv.exists():
                    df_clean = pd.read_csv(cached_csv, index_col=0, parse_dates=True)
                    _, presence = load_redd_house(h_dir, h_id, appliances=appliances)
                    print(f"Loaded cached REDD House {h_id}: {len(df_clean)} samples @ 6s ({len(df_clean)*6/3600/24:.1f} days)")
                else:
                    df_raw, presence = load_redd_house(h_dir, h_id, appliances=appliances)
                    df_clean = resample_and_clean(
                        df_raw,
                        sample_period_seconds=config.sample_period_seconds,
                        max_gap_fill_samples=config.max_gap_fill_samples,
                    )
                    df_clean.to_csv(cached_csv)
                    print(f"Saved & loaded REDD House {h_id}: {len(df_clean)} samples @ 6s ({len(df_clean)*6/3600/24:.1f} days)")
                house_dfs[h_id] = df_clean
                house_presences[h_id] = presence
            except Exception as e:
                print(f"Skipping {h_dir.name}: {e}")
    else:
        raise ValueError("Must specify --redd_dir to train on real REDD data.")

    if not house_dfs:
        raise RuntimeError("No datasets could be loaded.")

    held_out_id = config.held_out_house
    train_dfs: List[Tuple[pd.DataFrame, Dict[str, bool]]] = []
    val_dfs: List[Tuple[pd.DataFrame, Dict[str, bool]]] = []
    test_held_out_dfs: List[Tuple[pd.DataFrame, Dict[str, bool]]] = []

    for h_id, df in house_dfs.items():
        presence = house_presences.get(h_id, {app: True for app in appliances})
        if h_id == held_out_id:
            test_held_out_dfs.append((df, presence))
            print(f"Held-out Generalization House: House {h_id} ({len(df)} samples)")
        else:
            split_idx = int(len(df) * config.train_val_split_ratio)
            train_dfs.append((df.iloc[:split_idx], presence))
            val_dfs.append((df.iloc[split_idx:], presence))

    if not test_held_out_dfs:
        last_id = max(house_dfs.keys())
        print(f"Held-out house {held_out_id} not found. Reserving House {last_id} for cross-household eval.")
        test_held_out_dfs.append((house_dfs[last_id], house_presences[last_id]))
        train_dfs = []
        val_dfs = []
        for h_id, df in house_dfs.items():
            if h_id != last_id:
                split_idx = int(len(df) * config.train_val_split_ratio)
                train_dfs.append((df.iloc[:split_idx], house_presences[h_id]))
                val_dfs.append((df.iloc[split_idx:], house_presences[h_id]))

    # Compute normalization statistics exclusively on training splits
    combined_train_df = pd.concat([t[0] for t in train_dfs], axis=0) if train_dfs else list(house_dfs.values())[0]
    norm_params = compute_normalization_params(
        combined_train_df,
        appliances=appliances,
        thresholds=config.thresholds,
    )

    print("\n================ NORMALIZATION PARAMETERS GENERATION ================")
    print(f"[Verification] Freshly generated norm_params from {len(combined_train_df)} training samples.")
    print(f"[Verification] Aggregate Mains: mean = {norm_params.mains_mean:.2f} W, std = {norm_params.mains_std:.2f} W")
    for app in appliances:
        stats = norm_params.appliance_stats.get(app, {})
        print(f"  - {app:15s}: active_mean = {stats.get('active_mean', 0.0):.2f} W, "
              f"active_std = {stats.get('active_std', 1.0):.2f} W, threshold = {stats.get('threshold', 20.0):.1f} W")
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

    train_x = np.concatenate(x_train_list, axis=0) if x_train_list else np.zeros((0, config.window_length, 1))
    train_yp = np.concatenate(yp_train_list, axis=0) if yp_train_list else np.zeros((0, config.window_length, len(appliances)))
    train_yo = np.concatenate(yo_train_list, axis=0) if yo_train_list else np.zeros((0, config.window_length, len(appliances)))
    train_m = np.concatenate(m_train_list, axis=0) if m_train_list else np.zeros((0, len(appliances)))

    val_x = np.concatenate(x_val_list, axis=0) if x_val_list else np.zeros((0, config.window_length, 1))
    val_yp = np.concatenate(yp_val_list, axis=0) if yp_val_list else np.zeros((0, config.window_length, len(appliances)))
    val_yo = np.concatenate(yo_val_list, axis=0) if yo_val_list else np.zeros((0, config.window_length, len(appliances)))
    val_m = np.concatenate(m_val_list, axis=0) if m_val_list else np.zeros((0, len(appliances)))

    test_x = np.concatenate(x_test_list, axis=0) if x_test_list else np.zeros((0, config.window_length, 1))
    test_yp = np.concatenate(yp_test_list, axis=0) if yp_test_list else np.zeros((0, config.window_length, len(appliances)))
    test_yo = np.concatenate(yo_test_list, axis=0) if yo_test_list else np.zeros((0, config.window_length, len(appliances)))
    test_m = np.concatenate(m_test_list, axis=0) if m_test_list else np.zeros((0, len(appliances)))

    print(f"Generated windows -> Train: {len(train_x)}, Val: {len(val_x)}, Held-out Test: {len(test_x)}")

    train_dataset = NILMDataset(train_x, train_yp, train_yo, train_m)
    val_dataset = NILMDataset(val_x, val_yp, val_yo, val_m)
    test_dataset = NILMDataset(test_x, test_yp, test_yo, test_m)

    if return_raw_splits:
        return train_dataset, val_dataset, test_dataset, norm_params, train_dfs, house_dfs

    return train_dataset, val_dataset, test_dataset, norm_params


def compute_active_window_sampler(
    dataset: NILMDataset,
    appliances: List[str],
    boost_weight: float = 2.5,
    underrep_threshold: float = 0.25,
) -> Tuple[WeightedRandomSampler, pd.DataFrame]:
    """Computes active-window sampling weights for PyTorch WeightedRandomSampler.

    Fix 1: Start at 1.0, and add +4.0 for each target appliance (microwave, dishwasher,
    washing machine, or any under-represented appliance in this fold) that has at least
    one active timestep in that window.

    Validation and test sets must NEVER be sampled; they stay at their natural distribution.
    """
    y_onoff = dataset.y_onoff  # (N, window_length, num_appliances)
    num_samples = len(y_onoff)

    # Boolean mask: True if appliance k has at least one active timestep in window i
    is_active_window = (y_onoff == 1.0).any(dim=1)  # (N, num_appliances)
    raw_freqs = is_active_window.float().mean(dim=0).cpu().numpy()

    # Under-represented appliances in this fold (microwave, dishwasher, washing machine, or < 25%)
    underrep_indices = [
        i for i, app in enumerate(appliances)
        if app in ["microwave", "dishwasher", "washing_machine"] or raw_freqs[i] < underrep_threshold
    ]

    weights = torch.ones(num_samples, dtype=torch.float32)
    for idx in underrep_indices:
        weights += boost_weight * is_active_window[:, idx].float()

    total_weight = torch.sum(weights).item()

    rows = []
    for i, app in enumerate(appliances):
        raw_cnt = int(is_active_window[:, i].sum().item())
        raw_freq = float(raw_freqs[i])
        act_weight_sum = float(torch.sum(weights[is_active_window[:, i]]).item())
        eff_freq = act_weight_sum / total_weight if total_weight > 0 else raw_freq
        boost = eff_freq / raw_freq if raw_freq > 0 else 1.0
        rows.append({
            "Appliance": app,
            "Raw Active Windows": f"{raw_cnt}/{num_samples}",
            "Raw Freq (%)": f"{raw_freq * 100:.2f}%",
            "Effective Freq (%)": f"{eff_freq * 100:.2f}%",
            "Sampling Boost": f"{boost:.2f}x",
        })

    freq_df = pd.DataFrame(rows)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples,
        replacement=True,
    )
    return sampler, freq_df


def train_model(
    config: NILMConfig,
    train_dataset: NILMDataset,
    val_dataset: NILMDataset,
    norm_params: NormalizationParams,
    checkpoint_dir: str = "checkpoints",
    use_oversampling: bool = True,
    boost_weight: float = 2.5,
    resume: bool = True,
    on_epoch_end_callback: Optional[Any] = None,
    pretrained_weights_path: Optional[str] = None,
) -> Tuple[MultiApplianceNILM, Dict[str, Any]]:
    """Executes the training loop with active loss weighting, per-appliance tracking, and checkpointing."""
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    device = torch.device(config.device)
    print(f"Training on device: {device} with on_weight={config.on_weight}x")

    # Deterministic DataLoader RNG
    dl_gen = torch.Generator()
    dl_gen.manual_seed(42)

    # Compute deterministic run_hash for provenance tracking
    cfg_dict = asdict(config) if hasattr(config, "__dataclass_fields__") else vars(config)
    run_payload = {
        "seed": 42,
        "config": cfg_dict,
        "boost_weight": boost_weight,
        "use_oversampling": use_oversampling,
        "pretrained_weights_path": str(pretrained_weights_path) if pretrained_weights_path else None,
    }
    run_hash = hashlib.sha256(json.dumps(run_payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    print(f"[Provenance] Deterministic Run Hash: {run_hash}")

    if use_oversampling:
        train_sampler, freq_df = compute_active_window_sampler(
            dataset=train_dataset,
            appliances=config.appliances,
            boost_weight=boost_weight,
        )
        print("\n================ ACTIVE-WINDOW OVERSAMPLING FREQUENCY AUDIT ================")
        print(f"Weight Formulation: W_i = 1.0 + {boost_weight:.1f} * sum(I[rare_app_k active in window i])")
        print(freq_df.to_string(index=False))
        print("============================================================================\n")
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=train_sampler,
            drop_last=(len(train_dataset) > config.batch_size),
            worker_init_fn=seed_worker,
            generator=dl_gen,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=(len(train_dataset) > config.batch_size),
            worker_init_fn=seed_worker,
            generator=dl_gen,
        )

    # Validation loader is strictly unweighted at natural distribution
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=dl_gen,
    )

    model = MultiApplianceNILM(
        appliances=config.appliances,
        conv_filters=config.conv_filters,
        conv_kernels=config.conv_kernels,
        dropout=config.encoder_dropout,
        lstm_hidden=config.lstm_hidden_size,
        head_conv_filters=config.head_conv_filters,
        head_dense_dim=config.head_dense_dim,
    ).to(device)

    # Load pretrained weights if specified (for transfer learning / fine-tuning)
    if pretrained_weights_path is not None:
        p_path = Path(pretrained_weights_path)
        if not p_path.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found at: {p_path}")
        print(f"\n================ PRETRAINED WEIGHT TRANSFER ================")
        print(f"[Transfer Learning] Loading pretrained weights from: {p_path}")
        p_ckpt = torch.load(str(p_path), map_location=device)
        state_dict = p_ckpt.get("model_state_dict", p_ckpt)
        clean_state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
        model.load_state_dict(clean_state_dict)
        p_epoch = p_ckpt.get("epoch")
        p_val_loss = p_ckpt.get("val_loss")
        print(f"[Transfer Learning] Pretrained Checkpoint Verification: Epoch {p_epoch}, Tensor Count {len(clean_state_dict)}, Val Loss {p_val_loss}")
        print(f"[Transfer Learning] Successfully loaded {len(clean_state_dict)} tensor weights into full network.")
        print(f"===========================================================\n")

    # Multi-GPU DataParallel wrapping across all available CUDA GPUs
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        num_gpus = torch.cuda.device_count()
        gpu_names = [f"GPU {i} ({torch.cuda.get_device_name(i)})" for i in range(num_gpus)]
        print(f"\n================ MULTI-GPU ACCELERATION ================")
        print(f"[Multi-GPU] Detected {num_gpus} CUDA devices! Wrapping model with torch.nn.DataParallel.")
        for name in gpu_names:
            print(f"  -> {name}")
        print(f"  -> Batch size {config.batch_size} will be split across {num_gpus} GPUs ({config.batch_size // num_gpus} per GPU).")
        print(f"========================================================\n")
        model = torch.nn.DataParallel(model)

    criterion = MultiApplianceLoss(
        appliances=config.appliances,
        lambda_bce=config.lambda_loss,
        on_weight=config.on_weight,
        appliance_weights=config.appliance_loss_weights,
        use_focal_loss=getattr(config, "use_focal_loss", False),
        focal_gamma=getattr(config, "focal_gamma", 2.0),
        focal_alpha=getattr(config, "focal_alpha", 0.25),
        focal_appliances=getattr(config, "focal_appliances", None),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_reduce_factor,
        patience=config.lr_reduce_patience,
        min_lr=config.min_lr,
    )

    use_amp = (device.type == "cuda" and config.mixed_precision)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")
    patience_counter = 0

    # History dictionary tracking per-appliance curves
    history: Dict[str, Any] = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
    }
    for app in config.appliances:
        history[f"train_{app}_loss"] = []
        history[f"val_{app}_loss"] = []
        history[f"train_{app}_mse"] = []
        history[f"val_{app}_mse"] = []
        history[f"train_{app}_bce"] = []
        history[f"val_{app}_bce"] = []

    norm_params_dict = asdict(norm_params)
    norm_params.save_json(ckpt_path / "norm_params.json")

    start_epoch = 1
    latest_ckpt_file = ckpt_path / "latest_checkpoint.pt"
    if resume and latest_ckpt_file.exists():
        try:
            raw_model = model.module if hasattr(model, "module") else model
            ckpt_state_dict = ckpt_state["model_state_dict"]
            clean_state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in ckpt_state_dict.items()}
            raw_model.load_state_dict(clean_state_dict)
            if "optimizer_state_dict" in ckpt_state:
                optimizer.load_state_dict(ckpt_state["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt_state and ckpt_state["scheduler_state_dict"] is not None:
                scheduler.load_state_dict(ckpt_state["scheduler_state_dict"])
            saved_epoch = ckpt_state.get("epoch", 0)
            start_epoch = saved_epoch + 1
            best_val_loss = ckpt_state.get("val_loss", float("inf"))
            history_file = ckpt_path / "history.json"
            if history_file.exists():
                with open(history_file, "r") as f:
                    history = json.load(f)
            print(f"Successfully resumed from epoch {saved_epoch}. Next epoch: {start_epoch} (Best val_loss so far: {best_val_loss:.4f})")
        except Exception as e:
            print(f"Warning: Failed to resume from checkpoint ({e}). Starting from epoch 1.")
            start_epoch = 1

    print(f"Starting training for {config.epochs} epochs (from epoch {start_epoch})...")
    start_time = time.time()

    for epoch in range(start_epoch, config.epochs + 1):
        # --- Training phase ---
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        train_app_loss_sums = {app: 0.0 for app in config.appliances}
        train_app_mse_sums = {app: 0.0 for app in config.appliances}
        train_app_bce_sums = {app: 0.0 for app in config.appliances}

        for x_b, yp_b, yo_b, mask_b in train_loader:
            x_b = x_b.to(device)
            yp_b = yp_b.to(device)
            yo_b = yo_b.to(device)
            mask_b = mask_b.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                preds = model(x_b)
                loss, breakdown = criterion(
                    power_pred=preds["power"],
                    power_true=yp_b,
                    onoff_pred=preds["on_off"],
                    onoff_true=yo_b,
                    appliance_mask=mask_b,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()
            train_batches += 1
            for app in config.appliances:
                train_app_loss_sums[app] += breakdown.get(f"{app}_loss", 0.0)
                train_app_mse_sums[app] += breakdown.get(f"{app}_mse", 0.0)
                train_app_bce_sums[app] += breakdown.get(f"{app}_bce", 0.0)

        avg_train_loss = train_loss_sum / max(1, train_batches)

        # --- Validation phase ---
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        val_app_loss_sums = {app: 0.0 for app in config.appliances}
        val_app_mse_sums = {app: 0.0 for app in config.appliances}
        val_app_bce_sums = {app: 0.0 for app in config.appliances}

        with torch.no_grad():
            for x_b, yp_b, yo_b, mask_b in val_loader:
                x_b = x_b.to(device)
                yp_b = yp_b.to(device)
                yo_b = yo_b.to(device)
                mask_b = mask_b.to(device)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    preds = model(x_b)
                    loss, breakdown = criterion(
                        power_pred=preds["power"],
                        power_true=yp_b,
                        onoff_pred=preds["on_off"],
                        onoff_true=yo_b,
                        appliance_mask=mask_b,
                    )
                val_loss_sum += loss.item()
                val_batches += 1
                for app in config.appliances:
                    val_app_loss_sums[app] += breakdown.get(f"{app}_loss", 0.0)
                    val_app_mse_sums[app] += breakdown.get(f"{app}_mse", 0.0)
                    val_app_bce_sums[app] += breakdown.get(f"{app}_bce", 0.0)

        avg_val_loss = val_loss_sum / max(1, val_batches) if val_batches > 0 else avg_train_loss

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(avg_val_loss)

        history["train_loss"].append(round(avg_train_loss, 4))
        history["val_loss"].append(round(avg_val_loss, 4))
        history["lr"].append(current_lr)

        # Record per-appliance breakdowns
        app_summary_parts = []
        for app in config.appliances:
            tr_l = train_app_loss_sums[app] / max(1, train_batches)
            va_l = val_app_loss_sums[app] / max(1, val_batches) if val_batches > 0 else tr_l
            tr_mse = train_app_mse_sums[app] / max(1, train_batches)
            va_mse = val_app_mse_sums[app] / max(1, val_batches) if val_batches > 0 else tr_mse
            tr_bce = train_app_bce_sums[app] / max(1, train_batches)
            va_bce = val_app_bce_sums[app] / max(1, val_batches) if val_batches > 0 else tr_bce

            history[f"train_{app}_loss"].append(round(tr_l, 4))
            history[f"val_{app}_loss"].append(round(va_l, 4))
            history[f"train_{app}_mse"].append(round(tr_mse, 4))
            history[f"val_{app}_mse"].append(round(va_mse, 4))
            history[f"train_{app}_bce"].append(round(tr_bce, 4))
            history[f"val_{app}_bce"].append(round(va_bce, 4))

            app_summary_parts.append(f"{app[:4]}: tr={tr_l:.2f}/va={va_l:.2f}")

        print(
            f"Epoch [{epoch:02d}/{config.epochs:02d}] "
            f"Total Train: {avg_train_loss:.4f} | Total Val: {avg_val_loss:.4f} | LR: {current_lr:.1e} | "
            + " | ".join(app_summary_parts)
        )

        # Checkpoint: save latest
        save_checkpoint(
            ckpt_path / "latest_checkpoint.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            val_loss=avg_val_loss,
            norm_params_dict=norm_params_dict,
            appliances=config.appliances,
            extra_metadata={
                "on_weight": config.on_weight,
                "held_out_house": config.held_out_house,
                "appliance_loss_weights": config.appliance_loss_weights,
                "boost_weight": boost_weight,
                "run_hash": run_hash,
            },
            scheduler=scheduler,
            run_hash=run_hash,
        )

        # Checkpoint: save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            save_checkpoint(
                ckpt_path / "best_model.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=best_val_loss,
                norm_params_dict=norm_params_dict,
                appliances=config.appliances,
                extra_metadata={
                    "on_weight": config.on_weight,
                    "held_out_house": config.held_out_house,
                    "appliance_loss_weights": config.appliance_loss_weights,
                    "boost_weight": boost_weight,
                    "run_hash": run_hash,
                },
                scheduler=scheduler,
                run_hash=run_hash,
            )
            print(f"  --> Saved new best model (val_loss: {best_val_loss:.4f}, run_hash: {run_hash[:10]})")
            export_best_checkpoint(ckpt_path, epoch, best_val_loss, run_hash)
        else:
            patience_counter += 1

        # Per-epoch history persistence for durability against unexpected kernel resets
        with open(ckpt_path / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        # Optional per-epoch callback (e.g. sync to external/Kaggle dataset)
        if on_epoch_end_callback is not None:
            try:
                on_epoch_end_callback(epoch, avg_train_loss, avg_val_loss, ckpt_path)
            except Exception as e:
                print(f"Warning: on_epoch_end_callback failed ({e})")

        if patience_counter >= config.early_stopping_patience:
            print(f"Early stopping triggered after {epoch} epochs (patience={config.early_stopping_patience}).")
            break

    total_time = time.time() - start_time
    print(f"Training completed in {total_time / 60:.2f} minutes. Best Val Loss: {best_val_loss:.4f}")

    with open(ckpt_path / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train NILM Multi-Appliance Seq2Seq Model")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Processed CSV directory")
    parser.add_argument("--redd_dir", type=str, default="data/raw/redd", help="Raw REDD directory")
    parser.add_argument("--epochs", type=int, default=35, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lr_reduce_patience", type=int, default=4, help="Patience for ReduceLROnPlateau")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate for ReduceLROnPlateau")
    parser.add_argument("--on_weight", type=float, default=None, help="Active on-state upweight multiplier (default: appliance-specific)")
    parser.add_argument("--boost_weight", type=float, default=2.5, help="Sampling boost weight added for active rare appliances")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--held_out_house", type=int, default=1, help="Held out house ID for generalization")
    parser.add_argument("--early_stopping_patience", type=int, default=35, help="Early stopping patience")
    parser.add_argument("--no_oversampling", action="store_true", help="Disable active-window oversampling")
    parser.add_argument("--pretrained_weights_path", type=str, default=None, help="Path to pretrained model checkpoint")
    parser.add_argument("--no_evaluate", action="store_true", help="Skip post-training evaluation on held-out house")
    args = parser.parse_args()

    # Reproducibility seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    cfg_kwargs = dict(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_reduce_patience=args.lr_reduce_patience,
        min_lr=args.min_lr,
        held_out_house=args.held_out_house,
        checkpoint_dir=args.checkpoint_dir,
        early_stopping_patience=args.early_stopping_patience,
    )
    if args.on_weight is not None:
        cfg_kwargs["on_weight"] = args.on_weight

    config = NILMConfig(**cfg_kwargs)

    platform_desc = f"Local Apple Silicon ({config.device.upper()}) | Single-Device Execution" if config.device == "mps" else f"{config.device.upper()} Execution"
    print("================================================================================")
    print("CANONICAL TRANSFER LEARNING BENCHMARK RUN (REDD HELD-OUT HOUSE 2)")
    print(f"Platform: {platform_desc}")
    print("Training Engine: src/train.py")
    print(f"Pretrained Weights: {args.pretrained_weights_path}")
    print(f"Hyperparameters: lr={config.lr}, on_weight={config.on_weight}, boost_weight={args.boost_weight},")
    print(f"                 ReduceLROnPlateau(factor={config.lr_reduce_factor}, patience={config.lr_reduce_patience}, min_lr={config.min_lr}),")
    print(f"                 Uniform w_k=1.0, BCE Gating=Enabled, Max Epochs={config.epochs}, Early Stop Patience={config.early_stopping_patience}")
    print("================================================================================\n")

    train_ds, val_ds, test_ds, norm_params = prepare_datasets(
        config=config,
        data_dir=args.data_dir,
        redd_dir=args.redd_dir,
    )

    try:
        train_model(
            config=config,
            train_dataset=train_ds,
            val_dataset=val_ds,
            norm_params=norm_params,
            checkpoint_dir=args.checkpoint_dir,
            use_oversampling=not args.no_oversampling,
            boost_weight=args.boost_weight,
            pretrained_weights_path=args.pretrained_weights_path,
        )
    finally:
        best_ckpt = Path(args.checkpoint_dir) / "best_model.pt"
        if best_ckpt.exists():
            sha256_hash = hashlib.sha256(best_ckpt.read_bytes()).hexdigest()
            file_size = best_ckpt.stat().st_size
            print("\n================ CANONICAL CHECKPOINT SHA-256 ================")
            print(f"Checkpoint Path:     {best_ckpt.resolve()}")
            print(f"File Size:           {file_size:,} bytes")
            print(f"SHA-256 Hash:        {sha256_hash}")
            print("==============================================================\n")
        else:
            print(f"\n[Warning] Checkpoint best_model.pt not found in {args.checkpoint_dir}")

    if best_ckpt.exists() and not args.no_evaluate:
        from src.evaluate import run_evaluation
        run_evaluation(
            checkpoint_path=str(best_ckpt),
            data_dir=args.data_dir,
            redd_dir=args.redd_dir,
            device=config.device,
            held_out_house=config.held_out_house,
            output_path=str(Path(args.checkpoint_dir) / "eval_results.json"),
        )


if __name__ == "__main__":
    main()
