"""Post-hoc decision-threshold sweep for NILM on/off classification heads.

Evaluates an existing checkpoint on the cross-household held-out House 2 test set
at swept threshold values (0.02..0.80), computing:
  - Timestep-level: Precision, Recall, F1, F2, F3
  - Event-level: Event Recall (fraction of contiguous ground-truth ON events
    that overlap with at least one predicted ON timestep)
  - NDE (with hard-gated power predictions at each threshold)

No retraining, no weight changes, no gating logic changes.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import NILMConfig
from src.data_pipeline import NILMDataset, NormalizationParams
from src.model import MultiApplianceNILM
from src.utils import compute_nde, compute_mae, compute_sae


# ─── Threshold grid ───────────────────────────────────────────────────────────
def build_threshold_grid() -> List[float]:
    """0.02 to 0.10 step 0.02, 0.10 to 0.50 step 0.05, 0.50 to 0.80 step 0.05."""
    thresholds = []
    # Fine steps in low range
    t = 0.02
    while t < 0.10 - 1e-9:
        thresholds.append(round(t, 4))
        t += 0.02
    # Medium steps
    t = 0.10
    while t <= 0.80 + 1e-9:
        thresholds.append(round(t, 4))
        t += 0.05
    return thresholds


# ─── Metric computation ──────────────────────────────────────────────────────
def compute_fbeta(precision: float, recall: float, beta: float) -> float:
    """F-beta score: (1+beta^2) * P * R / (beta^2 * P + R)."""
    if precision + recall < 1e-12:
        return 0.0
    b2 = beta ** 2
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def compute_event_recall(
    o_true_bin: np.ndarray,
    o_pred_bin: np.ndarray,
) -> Tuple[float, int, int]:
    """Event-level recall: fraction of ground-truth ON events overlapping a prediction.

    An 'event' = contiguous run of 1s in o_true_bin (1-D flattened).
    An event is 'detected' if ANY timestep within it has o_pred_bin == 1.

    Returns: (event_recall, n_detected, n_total_events)
    """
    true_flat = o_true_bin.flatten().astype(int)
    pred_flat = o_pred_bin.flatten().astype(int)

    n_events = 0
    n_detected = 0
    in_event = False
    event_detected = False

    for i in range(len(true_flat)):
        if true_flat[i] == 1:
            if not in_event:
                # Start of new event
                in_event = True
                event_detected = False
                n_events += 1
            if pred_flat[i] == 1:
                event_detected = True
        else:
            if in_event:
                # End of event
                if event_detected:
                    n_detected += 1
                in_event = False
    # Handle event that runs to end of array
    if in_event and event_detected:
        n_detected += 1

    if n_events == 0:
        return float("nan"), 0, 0
    return n_detected / n_events, n_detected, n_events


def sweep_threshold_for_appliance(
    o_true_bin: np.ndarray,
    o_pred_prob: np.ndarray,
    p_true_watts: np.ndarray,
    p_pred_watts_raw: np.ndarray,
    thresholds: List[float],
) -> List[Dict[str, Any]]:
    """Evaluates all metrics at each threshold for a single appliance.

    Args:
        o_true_bin:  Ground-truth on/off binary (flattened or N,T)
        o_pred_prob: Model sigmoid probabilities (same shape)
        p_true_watts: Ground-truth power in Watts (same shape)
        p_pred_watts_raw: Model predicted power in Watts BEFORE hard gating
        thresholds: list of thresholds to sweep
    """
    o_true_flat = (o_true_bin.flatten() >= 0.5).astype(int)
    o_pred_flat = o_pred_prob.flatten()
    p_true_flat = p_true_watts.flatten()
    p_pred_flat = p_pred_watts_raw.flatten()

    results = []
    for thr in thresholds:
        pred_bin = (o_pred_flat >= thr).astype(int)

        tp = int(np.sum((pred_bin == 1) & (o_true_flat == 1)))
        fp = int(np.sum((pred_bin == 1) & (o_true_flat == 0)))
        fn = int(np.sum((pred_bin == 0) & (o_true_flat == 1)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = compute_fbeta(precision, recall, beta=1.0)
        f2 = compute_fbeta(precision, recall, beta=2.0)
        f3 = compute_fbeta(precision, recall, beta=3.0)

        # Event recall
        event_recall, n_detected, n_events = compute_event_recall(
            o_true_flat.reshape(o_true_bin.shape),
            pred_bin.reshape(o_pred_prob.shape),
        )

        # NDE with hard-gated power at this threshold
        p_gated = p_pred_flat * pred_bin
        nde = compute_nde(p_true_flat, p_gated)

        # MAE
        mae = float(np.mean(np.abs(p_true_flat - p_gated)))

        results.append({
            "threshold": thr,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "f2": round(f2, 4),
            "f3": round(f3, 4),
            "event_recall": round(event_recall, 4) if not np.isnan(event_recall) else "NaN",
            "events_detected": n_detected,
            "events_total": n_events,
            "nde": round(nde, 4),
            "mae_watts": round(mae, 2),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })

    return results


# ─── Main sweep function ─────────────────────────────────────────────────────
def run_threshold_sweep(
    checkpoint_path: str,
    data_dir: str = "data/processed",
    redd_dir: str = "data/raw/redd",
    device: str = "cpu",
    held_out_house: int = 2,
    target_appliances: Optional[List[str]] = None,
    batch_size: int = 128,
):
    """Run full threshold sweep on a checkpoint's held-out test set."""

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # ── Load checkpoint ──────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"THRESHOLD SWEEP — {ckpt_path}")
    print(f"{'='*80}")

    ckpt = torch.load(str(ckpt_path), map_location=device)
    appliances = ckpt.get("appliances", ["fridge", "microwave", "dishwasher", "washing_machine"])
    target_appliances = target_appliances or ["fridge", "dishwasher"]

    print(f"Checkpoint epoch: {ckpt.get('epoch', '?')}")
    print(f"Checkpoint val_loss: {ckpt.get('val_loss', '?')}")
    print(f"All appliances in checkpoint: {appliances}")
    print(f"Sweeping thresholds for: {target_appliances}")
    print(f"Held-out house: {held_out_house}")

    # ── Norm params ──────────────────────────────────────────────────────
    norm_dict = ckpt.get("norm_params")
    if norm_dict:
        norm_params = NormalizationParams(
            mains_mean=norm_dict["mains_mean"],
            mains_std=norm_dict["mains_std"],
            appliance_stats=norm_dict["appliance_stats"],
        )
        print(f"Norm params: from checkpoint (mains mean={norm_params.mains_mean:.2f} W)")
    else:
        norm_file = ckpt_path.parent / "norm_params.json"
        norm_params = NormalizationParams.load_json(norm_file)
        print(f"Norm params: from {norm_file}")

    # ── Load model ───────────────────────────────────────────────────────
    model = MultiApplianceNILM(appliances=appliances)
    state_dict = ckpt["model_state_dict"]
    clean_sd = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(clean_sd)
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

    # ── Prepare test dataset (held-out house only) ───────────────────────
    config = NILMConfig(appliances=appliances, device=device, held_out_house=held_out_house)
    from src.train import prepare_datasets
    _, _, test_ds, _ = prepare_datasets(config, data_dir=data_dir, redd_dir=redd_dir)
    print(f"Test dataset: {len(test_ds)} windows from House {held_out_house}")

    # ── Collect raw predictions ──────────────────────────────────────────
    print("\nRunning inference on test set...")
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    power_preds_list = []
    power_trues_list = []
    onoff_preds_list = []
    onoff_trues_list = []

    with torch.no_grad():
        for batch in loader:
            x_b = batch[0].to(device)
            yp_b = batch[1]
            yo_b = batch[2]

            preds = model(x_b)
            power_preds_list.append(preds["power"].cpu().numpy())
            power_trues_list.append(yp_b.numpy())
            onoff_preds_list.append(preds["on_off"].cpu().numpy())
            onoff_trues_list.append(yo_b.numpy())

    p_pred = np.concatenate(power_preds_list, axis=0)
    p_true = np.concatenate(power_trues_list, axis=0)
    o_pred = np.concatenate(onoff_preds_list, axis=0)
    o_true = np.concatenate(onoff_trues_list, axis=0)

    print(f"Inference complete: {p_pred.shape[0]} windows, {p_pred.shape[1]} timesteps/window")

    # ── Build threshold grid ─────────────────────────────────────────────
    thresholds = build_threshold_grid()
    print(f"Threshold grid ({len(thresholds)} values): {thresholds}")

    # ── Sweep each target appliance ──────────────────────────────────────
    all_results = {}
    for app in target_appliances:
        if app not in appliances:
            print(f"\n⚠ Appliance '{app}' not in checkpoint — skipping")
            continue

        i = appliances.index(app)
        app_stats = norm_params.appliance_stats.get(app, {})
        app_threshold_watts = app_stats.get("threshold", 20.0)

        # Extract per-appliance arrays
        p_pred_norm = p_pred[..., i]
        p_true_norm = p_true[..., i]
        p_pred_watts = norm_params.denormalize_appliance(p_pred_norm, app)
        p_true_watts = norm_params.denormalize_appliance(p_true_norm, app)

        o_pred_prob = o_pred[..., i]
        o_true_bin = o_true[..., i]

        active_count = int(np.sum(p_true_watts >= app_threshold_watts))

        print(f"\n{'='*80}")
        print(f"APPLIANCE: {app.upper()}")
        print(f"  Power threshold: {app_threshold_watts} W")
        print(f"  Active samples (>= threshold): {active_count}")
        print(f"  On/off sigmoid stats: min={o_pred_prob.min():.4f}, "
              f"max={o_pred_prob.max():.4f}, mean={o_pred_prob.mean():.4f}, "
              f"median={np.median(o_pred_prob):.4f}")
        print(f"  Ground-truth on-fraction: {o_true_bin.mean():.4f}")

        # Count ground-truth events
        _, _, n_events = compute_event_recall(o_true_bin, np.ones_like(o_true_bin))
        print(f"  Ground-truth contiguous ON events: {n_events}")
        if app == "fridge":
            print(f"  ⚠ NOTE ON FRIDGE EVENT COUNT ({n_events} events):")
            print(f"    There is NO minimum gap merging or duration filtering in this segmentation logic.")
            print(f"    Every 0->1 transition in the binary ground-truth label (threshold=50W at 6s) counts as a new event.")
            print(f"    Because the 6s power signal flickers across the 50W boundary during active compressor cycles,")
            print(f"    this yields an inflated count (~14.5s average duration) rather than true macro-cycles.")
        print(f"{'='*80}")

        # Print header
        print(f"\n{'Thresh':>6s} | {'Prec':>7s} {'Recall':>7s} {'F1':>7s} "
              f"{'F2':>7s} {'F3':>7s} | {'EvtRecall':>9s} {'EvtDet':>6s}/{'':<5s} "
              f"| {'NDE':>7s} {'MAE(W)':>7s} | {'TP':>8s} {'FP':>8s} {'FN':>8s}")
        print("-" * 115)

        sweep_results = sweep_threshold_for_appliance(
            o_true_bin=o_true_bin,
            o_pred_prob=o_pred_prob,
            p_true_watts=p_true_watts,
            p_pred_watts_raw=p_pred_watts,
            thresholds=thresholds,
        )

        for r in sweep_results:
            evt_str = f"{r['event_recall']}" if r['event_recall'] != "NaN" else "  NaN  "
            print(f"  {r['threshold']:>4.2f} | "
                  f"{r['precision']:>7.4f} {r['recall']:>7.4f} {r['f1']:>7.4f} "
                  f"{r['f2']:>7.4f} {r['f3']:>7.4f} | "
                  f"{evt_str:>9s} {r['events_detected']:>5d}/{r['events_total']:<5d} "
                  f"| {r['nde']:>7.4f} {r['mae_watts']:>7.2f} | "
                  f"{r['tp']:>8d} {r['fp']:>8d} {r['fn']:>8d}")

        all_results[app] = sweep_results

        # ── Find optimal thresholds ──────────────────────────────────────
        print(f"\n--- {app.upper()} OPTIMAL THRESHOLDS ---")
        best_f1 = max(sweep_results, key=lambda r: r["f1"])
        best_f2 = max(sweep_results, key=lambda r: r["f2"])
        best_f3 = max(sweep_results, key=lambda r: r["f3"])
        best_nde = min(sweep_results, key=lambda r: r["nde"])

        print(f"  Best F1 = {best_f1['f1']:.4f} at threshold = {best_f1['threshold']:.2f} "
              f"(P={best_f1['precision']:.4f}, R={best_f1['recall']:.4f})")
        print(f"  Best F2 = {best_f2['f2']:.4f} at threshold = {best_f2['threshold']:.2f} "
              f"(P={best_f2['precision']:.4f}, R={best_f2['recall']:.4f})")
        print(f"  Best F3 = {best_f3['f3']:.4f} at threshold = {best_f3['threshold']:.2f} "
              f"(P={best_f3['precision']:.4f}, R={best_f3['recall']:.4f})")
        print(f"  Best NDE = {best_nde['nde']:.4f} at threshold = {best_nde['threshold']:.2f}")

        if app == "dishwasher":
            print(f"\n  ⚠ NOTE: These optimal thresholds are selected on the same House 2 test")
            print(f"    slice used for evaluation — NOT a separate calibration split. This is a")
            print(f"    known limitation; results represent an upper bound, not a generalization claim.")

    return all_results


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-hoc threshold sweep for NILM on/off heads")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--redd_dir", type=str, default="data/raw/redd")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--held_out_house", type=int, default=2)
    parser.add_argument("--appliances", nargs="+", default=["fridge", "dishwasher"],
                        help="Appliances to sweep (default: fridge dishwasher)")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    run_threshold_sweep(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        redd_dir=args.redd_dir,
        device=args.device,
        held_out_house=args.held_out_house,
        target_appliances=args.appliances,
        batch_size=args.batch_size,
    )
