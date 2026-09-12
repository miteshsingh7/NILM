"""Leave-One-House-Out Cross-Validation (LOHO-CV) for UK-DALE Disaggregated Dataset.

Runs 5-fold leave-one-house-out CV across UK-DALE houses 1-5 under exact matching settings:
- Soft gating enabled during training.
- on_weight = 8.0x active-state upweighting.
- Uniform appliance loss weights (w_k = 1.0).
- Active-window oversampling with PyTorch WeightedRandomSampler and boost_weight = 2.5.
- 35 epochs per fold.
- Unweighted val/test splits.
- Pre-training active-sample audit per fold.
- Explicit evaluation stdout per fold.
- Final aggregate summary table (per-house rows + Aggregate Mean + Aggregate Range).
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import torch

from src.config import NILMConfig
from src.data_ukdale import (
    build_ukdale_coverage_table,
    load_ukdale_house,
    prepare_ukdale_datasets,
    print_ukdale_active_sample_audit,
    resample_and_clean,
)
from src.evaluate import run_evaluation
from src.train import train_model


def run_single_fold_ukdale(
    fold: int,
    epochs: int = 35,
    batch_size: int = 128,
    lr: float = 1e-3,
    on_weight: float = 8.0,
    base_checkpoint_dir: str = "checkpoints_ukdale",
    data_dir: str = "data/processed",
    ukdale_dir: str = "data/raw/ukdale",
    device: Optional[str] = None,
    use_oversampling: bool = True,
    boost_weight: float = 2.5,
) -> Dict[str, Any]:
    """Runs training and explicit evaluation for a single UK-DALE LOHO-CV fold."""
    ckpt_dir = Path(base_checkpoint_dir) / f"fold_{fold}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    config = NILMConfig(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        on_weight=on_weight,
        held_out_house=fold,
        checkpoint_dir=str(ckpt_dir),
    )
    if device:
        config.device = device

    print(f"\n################################################################################")
    print(f"          STARTING UK-DALE LOHO-CV FOLD {fold} (HELD-OUT: HOUSE {fold})         ")
    print(f"################################################################################\n")

    # 1. Dataset preparation
    train_ds, val_ds, test_ds, norm_params = prepare_ukdale_datasets(
        config=config,
        data_dir=data_dir,
        ukdale_dir=ukdale_dir,
    )

    # 2. Train model with active-window oversampling (boost_weight=2.5)
    model, history = train_model(
        config=config,
        train_dataset=train_ds,
        val_dataset=val_ds,
        norm_params=norm_params,
        checkpoint_dir=str(ckpt_dir),
        use_oversampling=use_oversampling,
        boost_weight=boost_weight,
    )

    # 3. Explicit evaluation
    best_ckpt_path = ckpt_dir / "best_model.pt"
    eval_output_path = ckpt_dir / "eval_results.json"
    print(f"\n================ EXPLICIT EVALUATION FOR FOLD {fold} ================\n")
    eval_results, _ = run_evaluation(
        checkpoint_path=str(best_ckpt_path),
        data_dir=data_dir,
        ukdale_dir=ukdale_dir,
        dataset_type="ukdale",
        output_path=str(eval_output_path),
        device=config.device,
        held_out_house=fold,
    )
    return eval_results


def build_ukdale_loho_summary_table(
    base_checkpoint_dir: str = "checkpoints_ukdale",
    appliances: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Builds the comprehensive UK-DALE LOHO-CV summary table across all 5 folds."""
    appliances = appliances or ["fridge", "microwave", "dishwasher", "washing_machine"]
    ckpt_base = Path(base_checkpoint_dir)

    fold_results: Dict[int, Dict[str, Any]] = {}
    for h in range(1, 6):
        result_file = ckpt_base / f"fold_{h}" / "eval_results.json"
        if result_file.exists():
            with open(result_file, "r") as f:
                fold_results[h] = json.load(f)

    if not fold_results:
        print(f"No fold eval_results.json found in {base_checkpoint_dir}/fold_*/")
        return pd.DataFrame()

    table_rows = []
    # Metrics to accumulate for mean and range
    metric_pools: Dict[str, List[float]] = {}
    for app in appliances:
        metric_pools[f"{app}_in_nde"] = []
        metric_pools[f"{app}_in_f1"] = []
        metric_pools[f"{app}_cr_nde"] = []
        metric_pools[f"{app}_cr_f1"] = []

    for h in range(1, 6):
        res = fold_results.get(h)
        row: Dict[str, Any] = {"Held-Out House": f"House {h}"}
        if res:
            in_dist = res.get("in_distribution", {})
            cross = res.get("cross_household", {})
            for app in appliances:
                in_nde = in_dist.get(app, {}).get("nde", np.nan)
                in_f1 = in_dist.get(app, {}).get("f1", np.nan)
                cr_nde = cross.get(app, {}).get("nde", np.nan)
                cr_f1 = cross.get(app, {}).get("f1", np.nan)

                # Format strings
                row[f"{app[:4]}_In_NDE"] = f"{in_nde:.4f}" if (in_nde is not None and not np.isnan(in_nde)) else "N/A"
                row[f"{app[:4]}_In_F1"] = f"{in_f1:.4f}" if (in_f1 is not None and not np.isnan(in_f1)) else "N/A"
                row[f"{app[:4]}_Cross_NDE"] = f"{cr_nde:.4f}" if (cr_nde is not None and not np.isnan(cr_nde)) else "N/A"
                row[f"{app[:4]}_Cross_F1"] = f"{cr_f1:.4f}" if (cr_f1 is not None and not np.isnan(cr_f1)) else "N/A"

                if in_nde is not None and not np.isnan(in_nde):
                    metric_pools[f"{app}_in_nde"].append(in_nde)
                if in_f1 is not None and not np.isnan(in_f1):
                    metric_pools[f"{app}_in_f1"].append(in_f1)
                if cr_nde is not None and not np.isnan(cr_nde):
                    metric_pools[f"{app}_cr_nde"].append(cr_nde)
                if cr_f1 is not None and not np.isnan(cr_f1):
                    metric_pools[f"{app}_cr_f1"].append(cr_f1)
        else:
            for app in appliances:
                row[f"{app[:4]}_In_NDE"] = "-"
                row[f"{app[:4]}_In_F1"] = "-"
                row[f"{app[:4]}_Cross_NDE"] = "-"
                row[f"{app[:4]}_Cross_F1"] = "-"
        table_rows.append(row)

    # Compute Mean Row
    mean_row: Dict[str, Any] = {"Held-Out House": "Aggregate Mean"}
    for app in appliances:
        in_ndes = metric_pools[f"{app}_in_nde"]
        in_f1s = metric_pools[f"{app}_in_f1"]
        cr_ndes = metric_pools[f"{app}_cr_nde"]
        cr_f1s = metric_pools[f"{app}_cr_f1"]

        mean_row[f"{app[:4]}_In_NDE"] = f"{np.mean(in_ndes):.4f}" if in_ndes else "N/A"
        mean_row[f"{app[:4]}_In_F1"] = f"{np.mean(in_f1s):.4f}" if in_f1s else "N/A"
        mean_row[f"{app[:4]}_Cross_NDE"] = f"{np.mean(cr_ndes):.4f}" if cr_ndes else "N/A"
        mean_row[f"{app[:4]}_Cross_F1"] = f"{np.mean(cr_f1s):.4f}" if cr_f1s else "N/A"
    table_rows.append(mean_row)

    # Compute Range Row (min - max)
    range_row: Dict[str, Any] = {"Held-Out House": "Aggregate Range"}
    for app in appliances:
        in_ndes = metric_pools[f"{app}_in_nde"]
        in_f1s = metric_pools[f"{app}_in_f1"]
        cr_ndes = metric_pools[f"{app}_cr_nde"]
        cr_f1s = metric_pools[f"{app}_cr_f1"]

        range_row[f"{app[:4]}_In_NDE"] = f"[{np.min(in_ndes):.2f}, {np.max(in_ndes):.2f}]" if in_ndes else "N/A"
        range_row[f"{app[:4]}_In_F1"] = f"[{np.min(in_f1s):.2f}, {np.max(in_f1s):.2f}]" if in_f1s else "N/A"
        range_row[f"{app[:4]}_Cross_NDE"] = f"[{np.min(cr_ndes):.2f}, {np.max(cr_ndes):.2f}]" if cr_ndes else "N/A"
        range_row[f"{app[:4]}_Cross_F1"] = f"[{np.min(cr_f1s):.2f}, {np.max(cr_f1s):.2f}]" if cr_f1s else "N/A"
    table_rows.append(range_row)

    summary_df = pd.DataFrame(table_rows)
    return summary_df


def main():
    parser = argparse.ArgumentParser(description="UK-DALE Leave-One-House-Out Cross-Validation (LOHO-CV)")
    parser.add_argument("--fold", type=int, default=None, choices=[1, 2, 3, 4, 5], help="Run specific fold (1-5)")
    parser.add_argument("--all_folds", action="store_true", help="Run all 5 folds sequentially")
    parser.add_argument("--summary_only", action="store_true", help="Print summary table from existing fold results")
    parser.add_argument("--epochs", type=int, default=35, help="Number of epochs per fold")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--on_weight", type=float, default=8.0, help="Active on-state upweight multiplier")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_ukdale", help="Base checkpoint directory")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Processed data directory")
    parser.add_argument("--ukdale_dir", type=str, default="data/raw/ukdale", help="Raw UK-DALE data directory")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument("--no_oversampling", action="store_true", help="Disable active-window oversampling")
    parser.add_argument("--boost_weight", type=float, default=2.5, help="Sampling boost weight added for active rare appliances")
    args = parser.parse_args()

    cfg = NILMConfig()

    # If only summary requested
    if args.summary_only:
        summary_df = build_ukdale_loho_summary_table(base_checkpoint_dir=args.checkpoint_dir)
        print("\n================ UK-DALE LEAVE-ONE-HOUSE-OUT CROSS-VALIDATION SUMMARY TABLE ================\n")
        print(summary_df.to_string(index=False))
        print("\n============================================================================================\n")
        return

    # Load house dfs for active-sample audit
    house_dfs = {}
    for h in range(1, 6):
        parquet_file = Path(args.data_dir) / f"ukdale_real_house_{h}.parquet"
        csv_file = Path(args.data_dir) / f"ukdale_real_house_{h}.csv"
        if parquet_file.exists():
            house_dfs[h] = pd.read_parquet(parquet_file)
        elif csv_file.exists():
            house_dfs[h] = pd.read_csv(csv_file, index_col=0, parse_dates=True)

    if args.fold is not None:
        folds_to_run = [args.fold]
    elif args.all_folds:
        folds_to_run = list(range(1, 6))
    else:
        print("Please specify either --fold <1-5> or --all_folds or --summary_only.")
        return

    for f in folds_to_run:
        if f in house_dfs:
            print_ukdale_active_sample_audit(house_dfs, cfg, fold=f)
        run_single_fold_ukdale(
            fold=f,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            on_weight=args.on_weight,
            base_checkpoint_dir=args.checkpoint_dir,
            data_dir=args.data_dir,
            ukdale_dir=args.ukdale_dir,
            device=args.device,
            use_oversampling=not args.no_oversampling,
            boost_weight=args.boost_weight,
        )

    summary_df = build_ukdale_loho_summary_table(base_checkpoint_dir=args.checkpoint_dir)
    print("\n================ UK-DALE LEAVE-ONE-HOUSE-OUT CROSS-VALIDATION SUMMARY TABLE ================\n")
    print(summary_df.to_string(index=False))
    print("\n============================================================================================\n")


if __name__ == "__main__":
    main()
