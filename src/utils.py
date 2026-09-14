"""Utility functions for energy metrics, cost calculation, serialization, and plotting."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch


def calculate_energy_kwh(watts: np.ndarray, step_seconds: int = 6) -> float:
    """Calculates total energy consumption in kilowatt-hours (kWh) from instantaneous watts.

    Formula: kWh = sum(watts) * (step_seconds / 3600) / 1000
    """
    total_watt_seconds = np.sum(np.clip(watts, 0.0, None)) * step_seconds
    kwh = total_watt_seconds / (3600.0 * 1000.0)
    return float(kwh)


def calculate_cost(kwh: float, tariff_per_kwh: float) -> float:
    """Calculates monetary cost for a given energy in kWh and tariff."""
    return float(kwh * tariff_per_kwh)


def compute_nde(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Normalized Disaggregation Error (NDE):
    NDE = sqrt(sum((y_pred - y_true)^2) / sum(y_true^2))
    """
    y_true_flat = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred_flat = np.asarray(y_pred, dtype=np.float64).flatten()

    denom = np.sum(y_true_flat ** 2)
    if denom < 1.0:
        # Ground truth has negligible power; report normalized RMSE against prediction std or RMSE
        rmse = float(np.sqrt(np.mean((y_pred_flat - y_true_flat) ** 2)))
        return round(rmse / (np.std(y_pred_flat) + 1.0), 4)

    numerator = np.sum((y_pred_flat - y_true_flat) ** 2)
    return float(np.sqrt(numerator / denom))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Mean Absolute Error in physical Watts."""
    return float(np.mean(np.abs(y_pred.flatten() - y_true.flatten())))


def compute_f1_score(
    y_true_onoff: np.ndarray,
    y_pred_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Computes Precision, Recall, and F1-score for binary on/off detection."""
    y_true_bin = (y_true_onoff.flatten() >= 0.5).astype(int)
    y_pred_bin = (y_pred_prob.flatten() >= threshold).astype(int)

    tp = np.sum((y_pred_bin == 1) & (y_true_bin == 1))
    fp = np.sum((y_pred_bin == 1) & (y_true_bin == 0))
    fn = np.sum((y_pred_bin == 0) & (y_true_bin == 1))
    tn = np.sum((y_pred_bin == 0) & (y_true_bin == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true_bin) if len(y_true_bin) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def compute_sae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    step_seconds: int = 6,
) -> float:
    """Computes Signal Aggregate Error (SAE):

    SAE = |E_pred - E_true| / E_true
    """
    e_true = calculate_energy_kwh(y_true, step_seconds)
    e_pred = calculate_energy_kwh(y_pred, step_seconds)
    if e_true <= 1e-6:
        return float(abs(e_pred - e_true))
    return float(abs(e_pred - e_true) / e_true)


def save_checkpoint(
    filepath: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    val_loss: float = float("inf"),
    norm_params_dict: Optional[Dict[str, Any]] = None,
    appliances: Optional[List[str]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
    scheduler: Optional[Any] = None,
    run_hash: Optional[str] = None,
) -> None:
    """Saves checkpoint with model state, optimizer, scheduler, metadata, and normalization parameters."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    raw_model = model.module if hasattr(model, "module") else model

    # Strict NaN/Inf guard before saving checkpoint to disk
    for p_name, p_tensor in raw_model.named_parameters():
        if torch.isnan(p_tensor).any() or torch.isinf(p_tensor).any():
            raise RuntimeError(f"FATAL: Attempted to save corrupted checkpoint containing NaN/Inf tensor: {p_name}")

    metadata = dict(extra_metadata) if extra_metadata else {}
    if run_hash:
        metadata["run_hash"] = run_hash

    state = {
        "epoch": epoch,
        "val_loss": val_loss,
        "run_hash": run_hash or metadata.get("run_hash", ""),
        "model_state_dict": raw_model.state_dict(),
        "appliances": appliances,
        "norm_params": norm_params_dict,
        "metadata": metadata,
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(state, str(filepath))


def load_checkpoint(
    filepath: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
    scheduler: Optional[Any] = None,
) -> Dict[str, Any]:
    """Loads checkpoint weights into model and optional optimizer / scheduler."""
    checkpoint = torch.load(str(filepath), map_location=device)
    raw_model = model.module if hasattr(model, "module") else model
    state_dict = checkpoint["model_state_dict"]
    clean_state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    raw_model.load_state_dict(clean_state_dict)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint
