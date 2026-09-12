"""Standalone module for NILM appliance on/off event detection and evaluation.

Extracts the binary state detection and event scoring logic embedded in the evaluation
pipeline into a clean, reusable interface compatible with any house or appliance pair.

Features:
- Exact scoring parity with Prompts 1 & 2 (compute_f1_score): Precision, Recall, F1, Accuracy.
- Supports continuous disaggregated power (Watts), binary on/off states, or probability outputs.
- Handles custom thresholds or defaults from NILM benchmark configurations.
- Flags near-zero active events (<100 samples) and unmonitored submeters.
- Works seamlessly with NumPy arrays, Pandas Series/DataFrames, and PyTorch tensors.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch

from src.config import DEFAULT_APPLIANCE_THRESHOLDS


def _to_numpy(data: Union[np.ndarray, pd.Series, torch.Tensor, List[float]]) -> np.ndarray:
    """Converts tensor, pandas series, or list into a flat float64 numpy array."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy().astype(np.float64).flatten()
    elif isinstance(data, pd.Series):
        return data.to_numpy(dtype=np.float64).flatten()
    elif isinstance(data, pd.DataFrame):
        return data.to_numpy(dtype=np.float64).flatten()
    elif isinstance(data, list):
        return np.asarray(data, dtype=np.float64).flatten()
    elif isinstance(data, np.ndarray):
        return data.astype(np.float64).flatten()
    else:
        return np.asarray(data, dtype=np.float64).flatten()


def detect_appliance_state(
    power_watts: Union[np.ndarray, pd.Series, torch.Tensor],
    threshold: float = 20.0,
) -> np.ndarray:
    """Converts a continuous power time series (in Watts) into binary ON (1) and OFF (0) states.

    Args:
        power_watts: Instantaneous power series in physical Watts.
        threshold: Power threshold in Watts above which the appliance is considered ON.

    Returns:
        Binary integer numpy array where 1 indicates ON and 0 indicates OFF.
    """
    arr = _to_numpy(power_watts)
    return (arr >= threshold).astype(np.int32)


def compute_event_metrics(
    y_true_binary: Union[np.ndarray, pd.Series, torch.Tensor],
    y_pred_binary: Union[np.ndarray, pd.Series, torch.Tensor],
) -> Dict[str, float]:
    """Computes sample-level binary classification metrics for on/off detection.

    Exact formula matching NILM benchmark scoring (Prompts 1 & 2):
        Precision = TP / (TP + FP)
        Recall    = TP / (TP + FN)
        F1-Score  = 2 * (Precision * Recall) / (Precision + Recall)
        Accuracy  = (TP + TN) / (TP + FP + FN + TN)

    Args:
        y_true_binary: Ground truth binary states (0 or 1).
        y_pred_binary: Predicted binary states (0 or 1).

    Returns:
        Dictionary containing precision, recall, f1, accuracy, tp, fp, fn, tn counts.
    """
    t = _to_numpy(y_true_binary).astype(np.int32)
    p = _to_numpy(y_pred_binary).astype(np.int32)

    assert len(t) == len(p), f"Length mismatch between ground truth ({len(t)}) and prediction ({len(p)})"

    tp = int(np.sum((p == 1) & (t == 1)))
    fp = int(np.sum((p == 1) & (t == 0)))
    fn = int(np.sum((p == 0) & (t == 1)))
    tn = int(np.sum((p == 0) & (t == 0)))

    total = len(t)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return {
        "precision": float(round(precision, 4)),
        "recall": float(round(recall, 4)),
        "f1": float(round(f1, 4)),
        "accuracy": float(round(accuracy, 4)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total_samples": total,
    }


def evaluate_appliance_events(
    y_pred: Union[np.ndarray, pd.Series, torch.Tensor, List[float]],
    y_true: Union[np.ndarray, pd.Series, torch.Tensor, List[float]],
    appliance: str = "fridge",
    power_threshold: Optional[float] = None,
    decision_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluates on/off event disaggregation for a single appliance.

    Handles:
    - Continuous disaggregated power (Watts): binarized at `power_threshold`.
    - Predicted probabilities in [0, 1]: binarized at `decision_threshold`.
    - Ground truth power (Watts): binarized at `power_threshold`.
    - Ground truth binary labels: used directly.

    Args:
        y_pred: Disaggregated power output (Watts) or predicted probability series.
        y_true: Ground truth power series (Watts) or binary labels (0/1).
        appliance: Appliance name (e.g. 'fridge', 'dishwasher', 'microwave', 'washing_machine').
        power_threshold: Power threshold in Watts. If None, loaded from DEFAULT_APPLIANCE_THRESHOLDS.
        decision_threshold: Probability decision threshold (default 0.5 matching Prompts 1 & 2).

    Returns:
        Dictionary containing precision, recall, f1, accuracy, sample counts, and status flags.
    """
    pred_arr = _to_numpy(y_pred)
    true_arr = _to_numpy(y_true)

    # Resolve physical power threshold in Watts
    if power_threshold is None:
        norm_app_key = appliance.lower().replace(" ", "_")
        p_thresh = DEFAULT_APPLIANCE_THRESHOLDS.get(norm_app_key, 20.0)
    else:
        p_thresh = float(power_threshold)

    # 1. Binarize Ground Truth
    # If values are in [0, 1] with unique values <= 2, treat as already binary
    unique_true = np.unique(true_arr[~np.isnan(true_arr)])
    if len(unique_true) <= 2 and np.all(np.isin(unique_true, [0, 1])):
        t_bin = (true_arr >= 0.5).astype(np.int32)
    else:
        t_bin = (true_arr >= p_thresh).astype(np.int32)

    # 2. Binarize Prediction
    # Detect if prediction is probability ([0.0, 1.0]) or continuous power in Watts
    pred_max = np.nanmax(pred_arr) if len(pred_arr) > 0 else 0.0
    pred_min = np.nanmin(pred_arr) if len(pred_arr) > 0 else 0.0
    is_prob = bool(pred_min >= 0.0 and pred_max <= 1.0 and len(np.unique(pred_arr)) > 2)

    if is_prob:
        p_bin = (pred_arr >= decision_threshold).astype(np.int32)
    elif len(np.unique(pred_arr)) <= 2 and np.all(np.isin(np.unique(pred_arr), [0, 1])):
        p_bin = (pred_arr >= 0.5).astype(np.int32)
    else:
        # Disaggregated continuous power in Watts
        p_bin = (pred_arr >= p_thresh).astype(np.int32)

    # 3. Compute Metrics
    metrics = compute_event_metrics(t_bin, p_bin)

    active_true = int(np.sum(t_bin == 1))
    active_pred = int(np.sum(p_bin == 1))

    # 4. Status Flagging (Near-zero activity or unmonitored)
    if active_true == 0 and np.all(true_arr == 0.0):
        status_flag = "UNMONITORED / ZERO GROUND TRUTH"
    elif active_true <= 100:
        status_flag = f"NEAR-ZERO ACTIVITY ({active_true} active samples)"
    else:
        status_flag = "ACTIVE"

    return {
        "appliance": appliance,
        "power_threshold_watts": p_thresh,
        "decision_threshold": decision_threshold,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "accuracy": metrics["accuracy"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tn": metrics["tn"],
        "active_samples_true": active_true,
        "active_samples_pred": active_pred,
        "total_samples": metrics["total_samples"],
        "status_flag": status_flag,
    }


def evaluate_disaggregation_events(
    predictions: Union[Dict[str, Any], pd.DataFrame],
    ground_truth: Union[Dict[str, Any], pd.DataFrame],
    thresholds: Optional[Dict[str, float]] = None,
    decision_threshold: float = 0.5,
) -> Dict[str, Dict[str, Any]]:
    """Evaluates multi-appliance event metrics across all appliances in a household.

    Args:
        predictions: Dictionary or DataFrame of predicted power traces or probabilities.
        ground_truth: Dictionary or DataFrame of ground truth power traces or labels.
        thresholds: Optional dictionary of power thresholds in Watts per appliance.
        decision_threshold: Probability threshold (default 0.5).

    Returns:
        Dictionary mapping appliance name -> event metrics dictionary.
    """
    thresholds = thresholds or {}

    # Standardize dictionary access
    if isinstance(predictions, pd.DataFrame):
        pred_dict = {col: predictions[col] for col in predictions.columns}
    else:
        pred_dict = dict(predictions)

    if isinstance(ground_truth, pd.DataFrame):
        true_dict = {col: ground_truth[col] for col in ground_truth.columns}
    else:
        true_dict = dict(ground_truth)

    results: Dict[str, Dict[str, Any]] = {}
    common_appliances = [k for k in pred_dict.keys() if k in true_dict]

    for app in common_appliances:
        p_th = thresholds.get(app, DEFAULT_APPLIANCE_THRESHOLDS.get(app, 20.0))
        results[app] = evaluate_appliance_events(
            y_pred=pred_dict[app],
            y_true=true_dict[app],
            appliance=app,
            power_threshold=p_th,
            decision_threshold=decision_threshold,
        )

    return results


def format_metrics_table(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Converts multi-appliance evaluation results dictionary to a presentation DataFrame."""
    rows = []
    for app, m in results.items():
        rows.append({
            "Appliance": app.replace("_", " ").title(),
            "Power Threshold (W)": f"{m['power_threshold_watts']:.1f} W",
            "F1-Score": m["f1"],
            "Precision": f"{m['precision'] * 100:.1f}%",
            "Recall": f"{m['recall'] * 100:.1f}%",
            "Accuracy": f"{m['accuracy'] * 100:.1f}%",
            "True Active Samples": m["active_samples_true"],
            "Pred Active Samples": m["active_samples_pred"],
            "Status": m["status_flag"],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=" * 70)
    print("EVENT DETECTION MODULE USAGE EXAMPLE")
    print("=" * 70)

    # 1. Example with synthetic/custom time series
    print("\n--- Example 1: Evaluating Continuous Disaggregated Power ---")
    np.random.seed(42)
    # 100 timesteps of fridge power
    # True fridge: turns on between t=20..50 at ~120W
    y_true_watts = np.zeros(100)
    y_true_watts[20:50] = 120.0 + np.random.normal(0, 5, 30)

    # Disaggregated model prediction: detects run between t=22..48 at ~110W
    y_pred_watts = np.zeros(100)
    y_pred_watts[22:48] = 110.0 + np.random.normal(0, 8, 26)

    fridge_metrics = evaluate_appliance_events(
        y_pred=y_pred_watts,
        y_true=y_true_watts,
        appliance="fridge",
        power_threshold=50.0,
    )
    print(f"Appliance: {fridge_metrics['appliance']}")
    print(f"Threshold: {fridge_metrics['power_threshold_watts']} W")
    print(f"Precision: {fridge_metrics['precision'] * 100:.1f}%")
    print(f"Recall:    {fridge_metrics['recall'] * 100:.1f}%")
    print(f"F1-Score:  {fridge_metrics['f1']:.4f}")
    print(f"TP={fridge_metrics['tp']}, FP={fridge_metrics['fp']}, FN={fridge_metrics['fn']}, TN={fridge_metrics['tn']}")

    # 2. Example with multiple appliances in a DataFrame
    print("\n--- Example 2: Multi-Appliance Evaluation ---")
    df_true = pd.DataFrame({
        "fridge": y_true_watts,
        "microwave": np.concatenate([np.zeros(80), np.ones(10) * 1200.0, np.zeros(10)]),
        "dishwasher": np.zeros(100),  # Inactive appliance
    })
    df_pred = pd.DataFrame({
        "fridge": y_pred_watts,
        "microwave": np.concatenate([np.zeros(82), np.ones(8) * 1150.0, np.zeros(10)]),
        "dishwasher": np.zeros(100),
    })

    multi_results = evaluate_disaggregation_events(df_pred, df_true)
    summary_df = format_metrics_table(multi_results)
    print(summary_df.to_string(index=False))
