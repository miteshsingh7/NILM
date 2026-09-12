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
from src.model import MultiApplianceNILM
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
) -> Dict[str, Dict[str, float]]:
    """Evaluates a dataset and computes physical-Watt NDE, MAE, On/Off F1, and Predict-Zero baseline."""
    model.eval()
    dev = torch.device(device)
    model.to(dev)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    power_preds_list = []
    power_trues_list = []
    onoff_preds_list = []
    onoff_trues_list = []

    with torch.no_grad():
        for batch in loader:
            x_b = batch[0].to(dev)
            yp_b = batch[1]
            yo_b = batch[2]

            preds = model(x_b)

            power_preds_list.append(preds["power"].cpu().numpy())
            power_trues_list.append(yp_b.numpy())
            onoff_preds_list.append(preds["on_off"].cpu().numpy())
            onoff_trues_list.append(yo_b.numpy())

    if not power_preds_list:
        return {}

    p_pred = np.concatenate(power_preds_list, axis=0)  # (N, 599, num_apps)
    p_true = np.concatenate(power_trues_list, axis=0)
    o_pred = np.concatenate(onoff_preds_list, axis=0)
    o_true = np.concatenate(onoff_trues_list, axis=0)

    results: Dict[str, Dict[str, float]] = {}

    for i, app in enumerate(appliances):
        p_pred_norm = p_pred[..., i]
        p_true_norm = p_true[..., i]

        # De-normalize predicted and ground truth power to physical Watts
        p_pred_watts = norm_params.denormalize_appliance(p_pred_norm, app)
        p_true_watts = norm_params.denormalize_appliance(p_true_norm, app)

        o_pred_prob = o_pred[..., i]
        o_true_bin = o_true[..., i]

        # Hard gate at eval time: zero out predicted power whenever on/off head predicts "off"
        p_pred_watts = p_pred_watts * (o_pred_prob >= 0.5)

        thresh = norm_params.appliance_stats.get(app, {}).get("threshold", 20.0)
        active_cnt = int(np.sum(p_true_watts >= thresh))

        if active_cnt > 0:
            nde = compute_nde(p_true_watts, p_pred_watts)
            zero_pred = np.zeros_like(p_true_watts)
            nde_zero_baseline = compute_nde(p_true_watts, zero_pred)
            beats_zero = bool(nde < nde_zero_baseline)
            clf_metrics = compute_f1_score(o_true_bin, o_pred_prob, threshold=0.5)
        else:
            # Appliance had zero active events in this test split or was unmetered
            nde = np.nan
            nde_zero_baseline = 1.0
            beats_zero = False
            clf_metrics = {"f1": np.nan, "precision": np.nan, "recall": np.nan, "accuracy": np.nan}

        mae = compute_mae(p_true_watts, p_pred_watts)
        sae = compute_sae(p_true_watts, p_pred_watts)

        results[app] = {
            "active_samples": active_cnt,
            "predict_zero_nde": round(nde_zero_baseline, 4),
            "nde": round(nde, 4) if not np.isnan(nde) else np.nan,
            "beats_zero_baseline": beats_zero,
            "mae_watts": round(mae, 2),
            "sae": round(sae, 4),
            "f1": round(clf_metrics["f1"], 4),
            "precision": round(clf_metrics["precision"], 4),
            "recall": round(clf_metrics["recall"], 4),
            "accuracy": round(clf_metrics["accuracy"], 4),
        }

    return results


def format_comparison_table(
    in_dist_metrics: Dict[str, Dict[str, float]],
    cross_house_metrics: Dict[str, Dict[str, float]],
    appliances: List[str],
) -> pd.DataFrame:
    """Formats side-by-side comparison between In-Distribution test, Cross-Household, and Predict-Zero."""
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

        in_mae = in_m.get("mae_watts", np.nan)
        cr_mae = cr_m.get("mae_watts", np.nan)

        rows.append({
            "Appliance": app,
            "Predict-Zero NDE": zero_nde,
            "In-Dist NDE": in_nde,
            "Cross-House NDE": cr_nde,
            "NDE Gap (Cross-In)": nde_gap,
            "In-Dist F1": in_f1,
            "Cross-House F1": cr_f1,
            "F1 Gap (Cross-In)": f1_gap,
            "In-Dist MAE (W)": in_mae,
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

    model = MultiApplianceNILM(appliances=appliances)
    state_dict = ckpt["model_state_dict"]
    clean_state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    if device == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    meta = ckpt.get("metadata") or ckpt.get("extra_metadata") or {}
    if held_out_house is None:
        held_out_house = meta.get("held_out_house", 1)
    config = NILMConfig(appliances=appliances, device=device, held_out_house=held_out_house)

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
            "In-Dist NDE": m.get("nde", np.nan),
            "In-Dist F1": m.get("f1", np.nan),
            "Precision": m.get("precision", np.nan),
            "Recall": m.get("recall", np.nan),
            "In-Dist MAE (W)": m.get("mae_watts", np.nan),
            "SAE": m.get("sae", np.nan),
        })
    in_dist_df = pd.DataFrame(in_dist_rows)

    cross_house_rows = []
    for app in appliances:
        m = cross_house_metrics.get(app, {})
        cross_house_rows.append({
            "Appliance": app,
            "Active Samples": m.get("active_samples", 0),
            "Cross-House NDE": m.get("nde", np.nan),
            "Cross-House F1": m.get("f1", np.nan),
            "Precision": m.get("precision", np.nan),
            "Recall": m.get("recall", np.nan),
            "Cross-House MAE (W)": m.get("mae_watts", np.nan),
            "SAE": m.get("sae", np.nan),
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
