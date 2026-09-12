"""Leave-One-House-Out Cross-Validation (LOHO-CV) for Multi-Appliance NILM.

Automates 6-fold leave-one-house-out CV across REDD houses 1-6:
- Computes active-sample counts per appliance for both training pool and held-out house.
- Applies Fix 1 active-window oversampling with PyTorch WeightedRandomSampler in each fold.
- Trains 35 epochs with uniform w_k=1.0 and on_weight=8.0x.
- Runs evaluate.py per fold, capturing unedited evaluation stdout.
- Produces grand summary table with In-Dist and Cross-House NDE/F1 per fold plus aggregate mean and range.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import torch

from src.config import NILMConfig
from src.evaluate import run_evaluation
from src.train import prepare_datasets, train_model


def print_active_sample_audit(house_dfs: Dict[int, pd.DataFrame], cfg: NILMConfig, fold: int) -> pd.DataFrame:
    """Prints active-sample counts per appliance for training pool vs held-out house."""
    appliances = cfg.appliances
    ho_df = house_dfs[fold]
    tr_houses = [h for h in range(1, 7) if h != fold]
    tr_df = pd.concat([house_dfs[h] for h in tr_houses], axis=0)

    print(f"\n================ FOLD {fold} ACTIVE SAMPLE AUDIT ================")
    print(f"Held-Out Generalization House: House {fold} ({len(ho_df):,} samples)")
    print(f"Training Pool Houses: {', '.join(f'House {h}' for h in tr_houses)} ({len(tr_df):,} samples)")
    print("----------------------------------------------------------------------------------------")

    rows = []
    for app in appliances:
        thresh = cfg.get_threshold(app)
        tr_cnt = int((tr_df[app] >= thresh).sum()) if app in tr_df.columns else 0
        tr_pct = tr_cnt / len(tr_df) * 100
        ho_cnt = int((ho_df[app] >= thresh).sum()) if app in ho_df.columns else 0
        ho_pct = ho_cnt / len(ho_df) * 100

        rows.append({
            "Appliance": app,
            "Training Pool Active Samples": f"{tr_cnt:,} / {len(tr_df):,} ({tr_pct:.2f}%)",
            "Held-Out House Active Samples": f"{ho_cnt:,} / {len(ho_df):,} ({ho_pct:.2f}%)",
        })

    audit_df = pd.DataFrame(rows)
    print(audit_df.to_string(index=False))
    print("========================================================================================\n")
    return audit_df


def run_single_fold(
    fold: int,
    epochs: int = 35,
    batch_size: int = 128,
    lr: float = 1e-3,
    on_weight: Union[float, Dict[str, float], None] = None,
    base_checkpoint_dir: str = "checkpoints",
    data_dir: str = "data/processed",
    redd_dir: str = "data/raw/redd",
    device: Optional[str] = None,
    use_oversampling: bool = True,
    boost_weight: float = 2.5,
) -> Dict[str, Any]:
    """Runs training and evaluation for a single LOHO-CV fold."""
    ckpt_dir = Path(base_checkpoint_dir) / f"fold_{fold}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cfg_kwargs = dict(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        held_out_house=fold,
        checkpoint_dir=str(ckpt_dir),
    )
    if on_weight is not None:
        cfg_kwargs["on_weight"] = on_weight
    config = NILMConfig(**cfg_kwargs)
    if device:
        config.device = device

    print(f"\n################################################################################")
    print(f"               STARTING LOHO-CV FOLD {fold} (HELD-OUT: HOUSE {fold})            ")
    print(f"################################################################################\n")

    # 1. Dataset preparation
    train_ds, val_ds, test_ds, norm_params = prepare_datasets(
        config=config,
        data_dir=data_dir,
        redd_dir=redd_dir,
    )

    # 2. Train model with active-window oversampling
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
    if best_ckpt_path.exists():
        import hashlib
        ckpt_bytes = best_ckpt_path.read_bytes()
        ckpt_sha256 = hashlib.sha256(ckpt_bytes).hexdigest()
        print(f"\n[Provenance] Fold {fold} Best Checkpoint SHA-256: {ckpt_sha256} ({len(ckpt_bytes):,} bytes)")
    print(f"\n================ EXPLICIT EVALUATION FOR FOLD {fold} ================\n")
    eval_results, _ = run_evaluation(
        checkpoint_path=str(best_ckpt_path),
        data_dir=data_dir,
        redd_dir=redd_dir,
        output_path=str(eval_output_path),
        device=config.device,
        held_out_house=fold,
    )
    return eval_results


def build_loho_summary_table(
    base_checkpoint_dir: str = "checkpoints",
    appliances: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Builds the comprehensive LOHO-CV summary table across all 6 folds."""
    appliances = appliances or ["fridge", "microwave", "dishwasher", "washing_machine"]
    ckpt_base = Path(base_checkpoint_dir)

    fold_results: Dict[int, Dict[str, Any]] = {}
    for h in range(1, 7):
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

    for h in range(1, 7):
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
                row[f"{app[:4]}_In_NDE"] = f"{in_nde:.4f}" if not np.isnan(in_nde) else "N/A"
                row[f"{app[:4]}_In_F1"] = f"{in_f1:.4f}" if not np.isnan(in_f1) else "N/A"
                row[f"{app[:4]}_Cross_NDE"] = f"{cr_nde:.4f}" if not np.isnan(cr_nde) else "N/A"
                row[f"{app[:4]}_Cross_F1"] = f"{cr_f1:.4f}" if not np.isnan(cr_f1) else "N/A"

                if not np.isnan(in_nde):
                    metric_pools[f"{app}_in_nde"].append(in_nde)
                if not np.isnan(in_f1):
                    metric_pools[f"{app}_in_f1"].append(in_f1)
                if not np.isnan(cr_nde):
                    metric_pools[f"{app}_cr_nde"].append(cr_nde)
                if not np.isnan(cr_f1):
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
    parser = argparse.ArgumentParser(description="Leave-One-House-Out Cross-Validation (LOHO-CV)")
    parser.add_argument("--fold", type=int, default=None, choices=[1, 2, 3, 4, 5, 6], help="Run specific fold (1-6)")
    parser.add_argument("--all_folds", action="store_true", help="Run all 6 folds sequentially")
    parser.add_argument("--summary_only", action="store_true", help="Print summary table from existing fold results")
    parser.add_argument("--epochs", type=int, default=35, help="Number of epochs per fold")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--on_weight", type=float, default=None, help="Active on-state upweight multiplier (default: appliance-specific)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Base checkpoint directory")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Processed data directory")
    parser.add_argument("--redd_dir", type=str, default="data/raw/redd", help="Raw REDD data directory")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument("--no_oversampling", action="store_true", help="Disable active-window oversampling")
    parser.add_argument("--boost_weight", type=float, default=2.5, help="Sampling boost weight added for active rare appliances")
    args = parser.parse_args()

    cfg = NILMConfig()

    # If only summary requested
    if args.summary_only:
        summary_df = build_loho_summary_table(base_checkpoint_dir=args.checkpoint_dir)
        print("\n================ LEAVE-ONE-HOUSE-OUT CROSS-VALIDATION SUMMARY TABLE ================\n")
        print(summary_df.to_string(index=False))
        print("\n====================================================================================\n")
        return

    # Load house dfs for active-sample audit
    house_dfs = {}
    for h in range(1, 7):
        csv_file = Path(args.data_dir) / f"redd_real_house_{h}.csv"
        if csv_file.exists():
            house_dfs[h] = pd.read_csv(csv_file, index_col=0, parse_dates=True)

    if args.fold is not None:
        folds_to_run = [args.fold]
    elif args.all_folds:
        folds_to_run = list(range(1, 7))
    else:
        print("Please specify either --fold <1-6> or --all_folds or --summary_only.")
        return

    for f in folds_to_run:
        if f in house_dfs:
            print_active_sample_audit(house_dfs, cfg, fold=f)
        run_single_fold(
            fold=f,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            on_weight=args.on_weight,
            base_checkpoint_dir=args.checkpoint_dir,
            data_dir=args.data_dir,
            redd_dir=args.redd_dir,
            device=args.device,
            use_oversampling=not args.no_oversampling,
            boost_weight=args.boost_weight,
        )

    summary_df = build_loho_summary_table(base_checkpoint_dir=args.checkpoint_dir)
    print("\n================ LEAVE-ONE-HOUSE-OUT CROSS-VALIDATION SUMMARY TABLE ================\n")
    print(summary_df.to_string(index=False))
    print("\n====================================================================================\n")


if __name__ == "__main__":
    main()
