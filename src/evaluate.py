"""Evaluation module for multi-appliance NILM.

Features:
- Normalized Disaggregation Error (NDE) on physical Watts.
- Predict-Zero NDE Baseline (1.0) for direct comparison against trivial non-detection.
- On/Off Precision, Recall, and F1-score.
- MAE (Watts) and SAE (Energy error).
- Cross-household generalization report comparing in-distribution vs unseen held-out house.
- Normalization consistency verification confirming checkpoint parameters are used.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import NILMConfig
from src.data_pipeline import (
    NILMDataset,
    NormalizationParams,
)
from src.model import MultiApplianceNILM, DecoupledTemporalNILM
from src.diagnostics import compute_phase0_diagnostics, format_phase0_diagnostic_table
from src.utils import (
    compute_f1_score,
    compute_mae,
    compute_nde,
    compute_sae,
    load_checkpoint,
)


def evaluate_dataset(
    model: MultiApplianceNILM,
    dataset: NILMDataset,
    norm_params: NormalizationParams,
    appliances: List[str],
    device: str = "cpu",
    batch_size: int = 128,
) -> Dict[str, Dict[str, Any]]:
    """Evaluates a dataset and computes physical-Watt NDE, MAE, On/Off F1, AP, Oracle F1, and Predict-Zero baseline."""
    model.eval()
    dev = torch.device(device)
    model.to(dev)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    power_preds_list = []
    power_trues_list = []
    onoff_preds_list = []
    onoff_trues_list = []
    mask_trues_list = []

    with torch.no_grad():
        for batch in loader:
            x_b = batch[0].to(dev)
            yp_b = batch[1]
            yo_b = batch[2]
            mask_b = batch[3] if len(batch) > 3 else torch.ones_like(yp_b)

            preds = model(x_b)

            power_preds_list.append(preds["power"].cpu().numpy())
            power_trues_list.append(yp_b.numpy())
            onoff_preds_list.append(preds["on_off"].cpu().numpy())
            onoff_trues_list.append(yo_b.numpy())
            mask_trues_list.append(mask_b.numpy())

    if not power_preds_list:
        return {}

    p_pred = np.concatenate(power_preds_list, axis=0)  # (N, 599, num_apps)
    p_true = np.concatenate(power_trues_list, axis=0)
    o_pred = np.concatenate(onoff_preds_list, axis=0)
    o_true = np.concatenate(onoff_trues_list, axis=0)
    m_true = np.concatenate(mask_trues_list, axis=0)

    results: Dict[str, Dict[str, Any]] = {}

    for i, app in enumerate(appliances):
        if m_true.ndim == 3:
            mask_app = m_true[..., i] > 0.5
        elif m_true.ndim == 2:
            mask_app = (m_true[:, i] > 0.5)[:, np.newaxis] & np.ones((len(m_true), p_pred.shape[1]), dtype=bool)
        else:
            mask_app = np.ones((len(p_pred), p_pred.shape[1]), dtype=bool)

        p_pred_norm = p_pred[..., i]
        p_true_norm = p_true[..., i]

        p_pred_watts = norm_params.denormalize_appliance(p_pred_norm, app)
        p_true_watts = norm_params.denormalize_appliance(p_true_norm, app)

        o_pred_prob = o_pred[..., i]
        o_true_bin = o_true[..., i]

        # Hard gate at eval time: zero out predicted power whenever on/off head predicts "off"
        p_pred_watts = p_pred_watts * (o_pred_prob >= 0.5)

        # Filter strictly by valid mask
        valid_p_true = p_true_watts[mask_app]
        valid_p_pred = p_pred_watts[mask_app]
        valid_o_true = o_true_bin[mask_app]
        valid_o_pred = o_pred_prob[mask_app]

        thresh = norm_params.appliance_stats.get(app, {}).get("threshold", 20.0)
        active_cnt = int(np.sum(valid_p_true >= thresh))
        valid_count = len(valid_p_true)

        if valid_count > 0 and active_cnt > 0:
            nde = compute_nde(valid_p_true, valid_p_pred)
            zero_pred = np.zeros_like(valid_p_true)
            nde_zero_baseline = compute_nde(valid_p_true, zero_pred)
            beats_zero = bool(nde < nde_zero_baseline)
            clf_metrics = compute_f1_score(valid_o_true, valid_o_pred, threshold=0.5)
            # Phase 0 diagnostics
            phase0_diag = compute_phase0_diagnostics(valid_o_true, valid_o_pred)
        else:
            nde = np.nan
            nde_zero_baseline = 1.0
            beats_zero = False
            clf_metrics = {"f1": np.nan, "precision": np.nan, "recall": np.nan, "accuracy": np.nan}
            phase0_diag = {
                "ap": np.nan,
                "oracle_f1": np.nan,
                "oracle_threshold": np.nan,
                "f1_at_50": np.nan,
                "ceiling_gap": np.nan,
                "active_timesteps": active_cnt,
                "total_timesteps": valid_count,
                "prevalence": 0.0,
            }

        mae = compute_mae(valid_p_true, valid_p_pred) if valid_count > 0 else np.nan
        sae = compute_sae(valid_p_true, valid_p_pred) if valid_count > 0 else np.nan

        results[app] = {
            "valid_samples": valid_count,
            "active_samples": active_cnt,
            "prevalence": round(phase0_diag.get("prevalence", 0.0), 6),
            "predict_zero_nde": round(nde_zero_baseline, 4),
            "nde": round(nde, 4) if not np.isnan(nde) else np.nan,
            "beats_zero_baseline": beats_zero,
            "mae_watts": round(mae, 2) if not np.isnan(mae) else np.nan,
            "sae": round(sae, 4) if not np.isnan(sae) else np.nan,
            "f1": round(clf_metrics["f1"], 4) if not np.isnan(clf_metrics["f1"]) else np.nan,
            "precision": round(clf_metrics["precision"], 4) if not np.isnan(clf_metrics["precision"]) else np.nan,
            "recall": round(clf_metrics["recall"], 4) if not np.isnan(clf_metrics["recall"]) else np.nan,
            "accuracy": round(clf_metrics["accuracy"], 4) if not np.isnan(clf_metrics["accuracy"]) else np.nan,
            "ap": phase0_diag.get("ap", np.nan),
            "oracle_f1": phase0_diag.get("oracle_f1", np.nan),
            "oracle_threshold": phase0_diag.get("oracle_threshold", np.nan),
            "ceiling_gap": phase0_diag.get("ceiling_gap", np.nan),
            "pr_curve": phase0_diag.get("pr_curve", {}),
        }

    return results


def format_comparison_table(
    in_dist_metrics: Dict[str, Dict[str, Any]],
    cross_house_metrics: Dict[str, Dict[str, Any]],
    appliances: List[str],
) -> pd.DataFrame:
    """Formats side-by-side comparison between In-Distribution test, Cross-Household, AP, and Oracle F1."""
    rows = []
    for app in appliances:
        in_m = in_dist_metrics.get(app, {})
        cr_m = cross_house_metrics.get(app, {})

        in_nde = in_m.get("nde", np.nan)
        cr_nde = cr_m.get("nde", np.nan)
        zero_nde = in_m.get("predict_zero_nde", 1.0)
        nde_gap = round(cr_nde - in_nde, 4) if not np.isnan(cr_nde) and not np.isnan(in_nde) else np.nan

        in_f1 = in_m.get("f1", np.nan)
        cr_f1 = cr_m.get("f1", np.nan)
        f1_gap = round(cr_f1 - in_f1, 4) if not np.isnan(cr_f1) and not np.isnan(in_f1) else np.nan

        in_ap = in_m.get("ap", np.nan)
        cr_ap = cr_m.get("ap", np.nan)
        cr_oracle = cr_m.get("oracle_f1", np.nan)
        cr_gap = cr_m.get("ceiling_gap", np.nan)

        in_mae = in_m.get("mae_watts", np.nan)
        cr_mae = cr_m.get("mae_watts", np.nan)

        rows.append({
            "Appliance": app,
            "In-Dist AP": in_ap,
            "Cross-House AP": cr_ap,
            "In-Dist F1": in_f1,
            "Cross-House F1": cr_f1,
            "Cross-House Oracle F1": cr_oracle,
            "Ceiling Gap": cr_gap,
            "Cross-House NDE": cr_nde,
            "Cross-House MAE (W)": cr_mae,
        })

    return pd.DataFrame(rows)


def run_evaluation(
    checkpoint_path: str = "checkpoints/best_model.pt",
    data_dir: str = "data/processed",
    redd_dir: Optional[str] = "data/raw/redd",
    ukdale_dir: Optional[str] = None,
    dataset_type: Optional[str] = None,
    output_path: Optional[str] = "checkpoints/eval_results.json",
    device: str = "cpu",
    held_out_house: Optional[int] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Runs complete evaluation on in-distribution validation and held-out test splits."""
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # Load checkpoint
    ckpt = torch.load(str(ckpt_file), map_location=device)
    appliances = ckpt.get("appliances", ["fridge", "microwave", "dishwasher", "washing_machine"])

    # Normalization parameter consistency check (Priority 3)
    norm_dict = ckpt.get("norm_params")
    if norm_dict:
        norm_source = f"checkpoint internal dict ({checkpoint_path})"
        norm_params = NormalizationParams(
            mains_mean=norm_dict["mains_mean"],
            mains_std=norm_dict["mains_std"],
            appliance_stats=norm_dict["appliance_stats"],
        )
    else:
        norm_file = ckpt_file.parent / "norm_params.json"
        assert norm_file.exists(), f"Normalization file not found: {norm_file}"
        norm_source = str(norm_file)
        norm_params = NormalizationParams.load_json(norm_file)

    print("\n================ NORMALIZATION PARAMETER VERIFICATION ================")
    print(f"[Verified] Using normalization statistics from: {norm_source}")
    print(f"[Verified] Mains: mean={norm_params.mains_mean:.2f} W, std={norm_params.mains_std:.2f} W")
    for app in appliances:
        stats = norm_params.appliance_stats.get(app, {})
        print(f"  - {app:15s}: active_mean={stats.get('active_mean', 0.0):.2f} W, "
              f"active_std={stats.get('active_std', 1.0):.2f} W, threshold={stats.get('threshold', 20.0):.1f} W")
    print("=======================================================================\n")

    state_dict = ckpt["model_state_dict"]
    clean_state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}

    # Auto-detect architecture, input channels, transition head, and normalization type from state_dict
    is_decoupled = any(k.startswith("lstms.") for k in clean_state_dict)
    has_running_stats = any("running_mean" in k for k in clean_state_dict)
    norm_type = "batchnorm" if has_running_stats else "groupnorm"

    in_channels = 1
    if "conv1.weight" in clean_state_dict:
        in_channels = clean_state_dict["conv1.weight"].shape[1]
    elif "encoder.conv1.weight" in clean_state_dict:
        in_channels = clean_state_dict["encoder.conv1.weight"].shape[1]

    predict_transitions = any("transition_classifier" in k for k in clean_state_dict)

    if is_decoupled:
        lstm_hidden = 48
        for k in clean_state_dict:
            if "weight_ih_l0" in k:
                lstm_hidden = clean_state_dict[k].shape[0] // 4
                break
        model = DecoupledTemporalNILM(
            appliances=appliances,
            in_channels=in_channels,
            lstm_hidden=lstm_hidden,
            norm_type=norm_type,
            predict_transitions=predict_transitions,
        )
    else:
        model = MultiApplianceNILM(
            appliances=appliances,
            in_channels=in_channels,
            norm_type=norm_type,
            predict_transitions=predict_transitions,
        )
    model.load_state_dict(clean_state_dict)
    if device == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    meta = ckpt.get("metadata") or ckpt.get("extra_metadata") or {}
    if held_out_house is None:
        held_out_house = meta.get("held_out_house", 1)
    config = NILMConfig(appliances=appliances, in_channels=in_channels, device=device, held_out_house=held_out_house)

    is_ukdale = (dataset_type == "ukdale") or (ukdale_dir is not None) or ("ukdale" in checkpoint_path.lower())
    if is_ukdale:
        ukdale_path = ukdale_dir or "data/raw/ukdale"
        from src.data_ukdale import prepare_ukdale_datasets
        train_ds, val_ds, test_ds, _ = prepare_ukdale_datasets(config, data_dir=data_dir, ukdale_dir=ukdale_path)
        all_houses = [1, 2, 3, 4, 5]
    else:
        from src.train import prepare_datasets
        train_ds, val_ds, test_ds, _ = prepare_datasets(config, data_dir=data_dir, redd_dir=redd_dir)
        all_houses = [1, 2, 3, 4, 5, 6]

    train_houses = [h for h in all_houses if h != held_out_house]
    print("\n================ DATASET SPLIT CONFIRMATION ================")
    print(f"Training Split Houses (80% train / 20% in-dist val): {', '.join(f'House {h}' for h in train_houses)}")
    print(f"Cross-Household Held-Out Test House (100% unseen)  : House {held_out_house}")
    print(f"Total Sliding Windows: Train={len(train_ds)}, In-Dist Val={len(val_ds)}, Cross-House Test={len(test_ds)}")
    print("============================================================\n")

    print("Evaluating In-Distribution Validation Set...")
    in_dist_metrics = evaluate_dataset(
        model=model,
        dataset=val_ds,
        norm_params=norm_params,
        appliances=appliances,
        device=device,
    )

    print("Evaluating Cross-Household Held-Out Test Set...")
    cross_house_metrics = evaluate_dataset(
        model=model,
        dataset=test_ds,
        norm_params=norm_params,
        appliances=appliances,
        device=device,
    )

    in_dist_rows = []
    for app in appliances:
        m = in_dist_metrics.get(app, {})
        in_dist_rows.append({
            "Appliance": app,
            "Active Samples": m.get("active_samples", 0),
            "In-Dist AP": m.get("ap", np.nan),
            "In-Dist F1": m.get("f1", np.nan),
            "Oracle F1": m.get("oracle_f1", np.nan),
            "Ceiling Gap": m.get("ceiling_gap", np.nan),
            "Precision": m.get("precision", np.nan),
            "Recall": m.get("recall", np.nan),
            "In-Dist NDE": m.get("nde", np.nan),
            "In-Dist MAE (W)": m.get("mae_watts", np.nan),
        })
    in_dist_df = pd.DataFrame(in_dist_rows)

    cross_house_rows = []
    for app in appliances:
        m = cross_house_metrics.get(app, {})
        cross_house_rows.append({
            "Appliance": app,
            "Active Samples": m.get("active_samples", 0),
            "Cross-House AP": m.get("ap", np.nan),
            "Cross-House F1": m.get("f1", np.nan),
            "Oracle F1": m.get("oracle_f1", np.nan),
            "Ceiling Gap": m.get("ceiling_gap", np.nan),
            "Precision": m.get("precision", np.nan),
            "Recall": m.get("recall", np.nan),
            "Cross-House NDE": m.get("nde", np.nan),
            "Cross-House MAE (W)": m.get("mae_watts", np.nan),
        })
    cross_house_df = pd.DataFrame(cross_house_rows)

    comp_df = format_comparison_table(in_dist_metrics, cross_house_metrics, appliances)

    full_results = {
        "norm_params_source": norm_source,
        "training_houses": train_houses,
        "held_out_house": held_out_house,
        "in_distribution": in_dist_metrics,
        "cross_household": cross_house_metrics,
        "comparison_table": comp_df.to_dict(orient="records"),
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(full_results, f, indent=2)
        print(f"\nSaved evaluation results to {output_path}")

    print(f"\n================ AGGREGATE IN-DISTRIBUTION VALIDATION ({', '.join(f'House {h}' for h in train_houses)}) ================\n")
    print(in_dist_df.to_string(index=False))
    print(f"\n================ AGGREGATE CROSS-HOUSEHOLD TEST (House {held_out_house} Held-Out) =========================\n")
    print(cross_house_df.to_string(index=False))
    print("\n================ SIDE-BY-SIDE GENERALIZATION COMPARISON REPORT =============================\n")
    print(comp_df.to_string(index=False))
    print("\n============================================================================================\n")

    return full_results, comp_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NILM Model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--redd_dir", type=str, default="data/raw/redd")
    parser.add_argument("--ukdale_dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None, choices=["redd", "ukdale"])
    parser.add_argument("--output", type=str, default="checkpoints/eval_results.json")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--held_out_house", type=int, default=None, help="Explicitly specify held-out house ID")
    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        redd_dir=args.redd_dir,
        ukdale_dir=args.ukdale_dir,
        dataset_type=args.dataset,
        output_path=args.output,
        device=args.device,
        held_out_house=args.held_out_house,
    )
