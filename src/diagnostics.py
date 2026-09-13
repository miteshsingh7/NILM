"""Diagnostic tools for NILM evaluation (Phase 0).

Computes:
- Per-house Precision-Recall (PR) curves
- Average Precision (AP) within-house
- Oracle-threshold F1 ceiling (best possible F1 across thresholds tau in [0.01, 0.90])
- Diagnostic reporting tables
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.utils import compute_f1_score


def compute_phase0_diagnostics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    mask: Optional[np.ndarray] = None,
    candidate_thresholds: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Computes Phase 0 diagnostics: PR curve, AP within-house, and Oracle F1 ceiling.

    Args:
        y_true: Ground truth binary state (0 or 1), 1D or flattened array.
        y_pred_prob: Model predicted probabilities in [0, 1], same shape as y_true.
        mask: Optional binary mask (1 for valid/metered, 0 for missing/unmetered).
        candidate_thresholds: Optional array of thresholds to evaluate for Oracle F1.

    Returns:
        Dictionary containing:
            - ap: Average precision score (within-house).
            - oracle_f1: Best achievable F1 across all thresholds.
            - oracle_threshold: The threshold tau achieving oracle_f1.
            - f1_at_50: F1 achieved at standard default threshold tau = 0.50.
            - ceiling_gap: oracle_f1 - f1_at_50 (headroom available from thresholding).
            - active_timesteps: Total count of active (1) timesteps.
            - total_timesteps: Total count of valid evaluated timesteps.
            - prevalence: Percentage of evaluated timesteps that are active.
            - pr_curve: Dict with precision, recall, and threshold arrays.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred_prob = np.asarray(y_pred_prob).ravel()

    if mask is not None:
        mask_flat = np.asarray(mask).ravel().astype(bool)
        y_true = y_true[mask_flat]
        y_pred_prob = y_pred_prob[mask_flat]

    total_timesteps = len(y_true)
    if total_timesteps == 0:
        return {
            "ap": np.nan,
            "oracle_f1": np.nan,
            "oracle_threshold": np.nan,
            "f1_at_50": np.nan,
            "ceiling_gap": np.nan,
            "active_timesteps": 0,
            "total_timesteps": 0,
            "prevalence": 0.0,
            "pr_curve": {"precision": [], "recall": [], "thresholds": []},
        }

    active_cnt = int(np.sum(y_true >= 0.5))
    prevalence = active_cnt / total_timesteps if total_timesteps > 0 else 0.0

    if active_cnt == 0:
        # No positive cycles observed in this split
        return {
            "ap": np.nan,
            "oracle_f1": np.nan,
            "oracle_threshold": np.nan,
            "f1_at_50": 0.0,
            "ceiling_gap": np.nan,
            "active_timesteps": 0,
            "total_timesteps": total_timesteps,
            "prevalence": 0.0,
            "pr_curve": {"precision": [], "recall": [], "thresholds": []},
        }

    # 1. Average Precision (AP)
    try:
        ap = float(average_precision_score(y_true, y_pred_prob))
    except Exception:
        ap = np.nan

    # 2. Precision-Recall Curve
    try:
        precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_pred_prob)
    except Exception:
        precisions, recalls, pr_thresholds = np.array([]), np.array([]), np.array([])

    # 3. Oracle F1 search
    if candidate_thresholds is None:
        candidate_thresholds = np.linspace(0.01, 0.90, 90)

    best_f1 = -1.0
    best_thresh = 0.50
    f1_50 = 0.0

    # Ensure 0.50 is included in evaluation
    eval_thresholds = np.unique(np.concatenate([candidate_thresholds, [0.50]]))

    for tau in eval_thresholds:
        m = compute_f1_score(y_true, y_pred_prob, threshold=float(tau))
        score = m["f1"]
        if abs(tau - 0.50) < 1e-4:
            f1_50 = score
        if score > best_f1:
            best_f1 = score
            best_thresh = float(tau)

    ceiling_gap = max(0.0, best_f1 - f1_50)

    return {
        "ap": round(ap, 4) if not np.isnan(ap) else np.nan,
        "oracle_f1": round(best_f1, 4),
        "oracle_threshold": round(best_thresh, 4),
        "f1_at_50": round(f1_50, 4),
        "ceiling_gap": round(ceiling_gap, 4),
        "active_timesteps": active_cnt,
        "total_timesteps": total_timesteps,
        "prevalence": round(prevalence, 6),
        "pr_curve": {
            "precision": precisions.tolist() if len(precisions) < 5000 else precisions[::10].tolist(),
            "recall": recalls.tolist() if len(recalls) < 5000 else recalls[::10].tolist(),
            "thresholds": pr_thresholds.tolist() if len(pr_thresholds) < 5000 else pr_thresholds[::10].tolist(),
        },
    }


def format_phase0_diagnostic_table(
    diagnostics_by_appliance: Dict[str, Dict[str, Any]],
    house_id: Union[int, str],
) -> pd.DataFrame:
    """Formats a concise markdown-ready DataFrame of Phase 0 metrics."""
    rows = []
    for app, diag in diagnostics_by_appliance.items():
        rows.append({
            "House": f"House {house_id}",
            "Appliance": app,
            "Prevalence": f"{diag.get('prevalence', 0.0)*100:.2f}% ({diag.get('active_timesteps', 0)}/{diag.get('total_timesteps', 0)})",
            "AP (In-House)": f"{diag.get('ap', np.nan):.4f}" if not np.isnan(diag.get("ap", np.nan)) else "N/A",
            "F1 (tau=0.50)": f"{diag.get('f1_at_50', np.nan):.4f}" if not np.isnan(diag.get("f1_at_50", np.nan)) else "N/A",
            "Oracle F1": f"{diag.get('oracle_f1', np.nan):.4f} (@ tau={diag.get('oracle_threshold', 0.5):.2f})" if not np.isnan(diag.get("oracle_f1", np.nan)) else "N/A",
            "Ceiling Gap": f"{diag.get('ceiling_gap', np.nan):+.4f}" if not np.isnan(diag.get("ceiling_gap", np.nan)) else "N/A",
        })
    return pd.DataFrame(rows)
