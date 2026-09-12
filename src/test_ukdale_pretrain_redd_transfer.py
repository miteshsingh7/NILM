"""Transfer Learning Experiment: UK-DALE 5-House Pretraining + REDD House 2 Fine-Tuning.

STAGE 1 — Pretrain on UK-DALE:
- All 5 UK-DALE houses (Houses 1-5, no held-out house).
- Normalization: Computed strictly on UK-DALE training data (norm_params_ukdale).
- Architecture: Shared Conv1D + BiLSTM encoder + 4 appliance dual-heads with soft gating.
- Loss: Uniform on_weight=8.0x BCE + gated MSE, boost_weight=2.5 oversampling.
- Saved to: checkpoints/ukdale_pretrained/best_model.pt.

STAGE 2 — Fine-tune on REDD:
- Load pretrained weights into the full network end-to-end.
- Training pool: REDD Houses 1, 3, 4, 5, 6 (House 2 held out for evaluation).
- Normalization: Computed strictly on REDD training split (norm_params_redd).
- Reduced Learning Rate: lr=1e-4 (one-tenth of 1e-3).
- Verified recipe: on_weight=8.0, uniform w_k=1.0, boost_weight=2.5, standard BCE (no focal loss).
- Fine-tune for up to 35 epochs with early stopping (patience=8).
- Final evaluation: Explicit evaluation on unseen REDD House 2.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import json
import numpy as np
import pandas as pd
import torch

from src.config import NILMConfig
from src.data_pipeline import NormalizationParams
from src.data_ukdale import prepare_ukdale_pretraining_datasets
from src.train import prepare_datasets, train_model
from src.evaluate import run_evaluation
from src.loho_cv import print_active_sample_audit


def print_normalization_audit(norm_ukdale: NormalizationParams, norm_redd: NormalizationParams):
    print("\n================================================================================")
    print("           NORMALIZATION STATISTICS AUDIT ACROSS EXPERIMENTAL STAGES            ")
    print("================================================================================")
    print("STAGE 1: UK-DALE Pretraining Statistics (230V / 50Hz UK System)")
    print(f"  - Mains: mean = {norm_ukdale.mains_mean:.2f} W, std = {norm_ukdale.mains_std:.2f} W")
    for app, stats in norm_ukdale.appliance_stats.items():
        print(f"  - {app:<15s}: active_mean = {stats['active_mean']:.2f} W, "
              f"active_std = {stats['active_std']:.2f} W, threshold = {stats['threshold']:.1f} W")
    print("  * Scope: Applied EXCLUSIVELY to Stage 1 UK-DALE pretraining windows.")

    print("\nSTAGE 2 & INFERENCE: REDD Fine-tuning & Evaluation Statistics (120V / 60Hz US System)")
    print(f"  - Mains: mean = {norm_redd.mains_mean:.2f} W, std = {norm_redd.mains_std:.2f} W")
    for app, stats in norm_redd.appliance_stats.items():
        print(f"  - {app:<15s}: active_mean = {stats['active_mean']:.2f} W, "
              f"active_std = {stats['active_std']:.2f} W, threshold = {stats['threshold']:.1f} W")
    print("  * Scope: Applied to REDD fine-tuning AND all downstream inference/evaluation on House 2.")
    print("  * Integrity check: The final checkpoint embeds REDD norm_params for test-time denormalization.")
    print("================================================================================\n")


def print_ukdale_active_counts(house_dfs: dict, cfg: NILMConfig):
    print("\n================ UK-DALE ACTIVE SAMPLES PER HOUSE AUDIT ================")
    rows = []
    for app in cfg.appliances:
        thresh = cfg.get_threshold(app)
        row = {"Appliance": app, "Threshold": f"{thresh:.0f} W"}
        total_active = 0
        total_samples = 0
        for h in range(1, 6):
            df = house_dfs.get(h)
            if df is not None and app in df.columns:
                cnt = int((df[app] >= thresh).sum())
                pct = cnt / len(df) * 100 if len(df) > 0 else 0.0
                row[f"House {h}"] = f"{cnt:,} ({pct:.2f}%)"
                total_active += cnt
                total_samples += len(df)
            else:
                row[f"House {h}"] = "0 (0.00%) / Unmetered"
        pct_tot = total_active / total_samples * 100 if total_samples > 0 else 0.0
        row["Total Active"] = f"{total_active:,} ({pct_tot:.2f}%)"
        rows.append(row)
    df_audit = pd.DataFrame(rows)
    print(df_audit.to_string(index=False))
    print("========================================================================\n")


def run_pretraining_and_finetuning():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using compute device: {device}")

    # =========================================================================
    # STAGE 1: UK-DALE PRETRAINING (ALL 5 HOUSES)
    # =========================================================================
    ukdale_ckpt_dir = Path("checkpoints/ukdale_pretrained")
    ukdale_ckpt_dir.mkdir(parents=True, exist_ok=True)
    ukdale_best_model = ukdale_ckpt_dir / "best_model.pt"

    config_ukdale = NILMConfig(
        epochs=10,
        batch_size=128,
        lr=1e-3,
        on_weight=8.0,
        lambda_loss=1.0,
        early_stopping_patience=4,
        checkpoint_dir=str(ukdale_ckpt_dir),
        device=device,
        dataset_name="ukdale",
        held_out_house=-1,  # No held-out house
        use_focal_loss=False,
    )

    print("\n################################################################################")
    print("             STAGE 1: PRETRAINING ON UK-DALE (ALL 5 HOUSES)                     ")
    print("################################################################################\n")

    # 1. Load UK-DALE Pretraining Datasets
    ukdale_train_ds, ukdale_val_ds, norm_ukdale, ukdale_house_dfs = prepare_ukdale_pretraining_datasets(
        config=config_ukdale,
        data_dir="data/processed",
        ukdale_dir="data/raw/ukdale",
    )

    # 2. Print UK-DALE Active Sample Audit
    print_ukdale_active_counts(ukdale_house_dfs, config_ukdale)

    # 3. Train on UK-DALE
    print(f"Starting UK-DALE pretraining for {config_ukdale.epochs} epochs...")
    model_ukdale, history_ukdale = train_model(
        config=config_ukdale,
        train_dataset=ukdale_train_ds,
        val_dataset=ukdale_val_ds,
        norm_params=norm_ukdale,
        checkpoint_dir=str(ukdale_ckpt_dir),
        use_oversampling=True,
        boost_weight=2.5,
        resume=True,  # Allows resuming if interrupted
    )

    print(f"\n✅ STAGE 1 COMPLETE: Saved UK-DALE pretrained checkpoint to {ukdale_best_model}\n")

    # =========================================================================
    # STAGE 2: REDD FINE-TUNING (FOLD 2 HELD-OUT)
    # =========================================================================
    redd_fold = 2
    redd_ckpt_dir = Path("checkpoints/redd_finetuned_fold2")
    redd_ckpt_dir.mkdir(parents=True, exist_ok=True)
    redd_best_model = redd_ckpt_dir / "best_model.pt"
    redd_eval_result = redd_ckpt_dir / "eval_results.json"

    # Reduced learning rate (1e-4) for fine-tuning
    config_redd = NILMConfig(
        epochs=35,
        batch_size=128,
        lr=1e-4,  # Reduced lr: 1/10th of 1e-3
        on_weight=8.0,
        lambda_loss=1.0,
        early_stopping_patience=8,
        checkpoint_dir=str(redd_ckpt_dir),
        device=device,
        dataset_name="redd",
        held_out_house=redd_fold,
        use_focal_loss=False,
    )

    print("\n################################################################################")
    print(f"     STAGE 2: REDD FINE-TUNING (UNSEEN HOUSE {redd_fold} HELD OUT, LR = 1e-4)     ")
    print("################################################################################\n")

    # 1. Prepare REDD Datasets
    redd_train_ds, redd_val_ds, redd_test_ds, norm_redd, redd_train_dfs, redd_house_dfs = prepare_datasets(
        config=config_redd,
        data_dir="data/processed",
        redd_dir="data/raw/redd",
        return_raw_splits=True,
    )

    # 2. Print REDD Active Sample Audit
    print_active_sample_audit(redd_house_dfs, config_redd, fold=redd_fold)

    # 3. Print Explicit Normalization Audit Comparing Stage 1 and Stage 2
    print_normalization_audit(norm_ukdale, norm_redd)

    # 4. Train with Pretrained Weights Loaded End-to-End
    print(f"Starting REDD end-to-end fine-tuning at lr={config_redd.lr:.1e} from UK-DALE weights...")
    model_redd, history_redd = train_model(
        config=config_redd,
        train_dataset=redd_train_ds,
        val_dataset=redd_val_ds,
        norm_params=norm_redd,
        checkpoint_dir=str(redd_ckpt_dir),
        use_oversampling=True,
        boost_weight=2.5,
        resume=False,  # Fresh start from pretrained weights
        pretrained_weights_path=str(ukdale_best_model),
    )

    # =========================================================================
    # EVALUATION ON HELD-OUT HOUSE 2
    # =========================================================================
    print(f"\n================ EXPLICIT EVALUATION FOR REDD FOLD {redd_fold} ================\n")
    eval_results, _ = run_evaluation(
        checkpoint_path=str(redd_best_model),
        data_dir="data/processed",
        redd_dir="data/raw/redd",
        dataset_type="redd",
        output_path=str(redd_eval_result),
        device=device,
        held_out_house=redd_fold,
    )

    # =========================================================================
    # HEAD-TO-HEAD COMPARISON TABLE
    # =========================================================================
    with open(redd_eval_result, "r") as f:
        res = json.load(f)

    cross = res.get("cross_household", {})

    baseline_b25 = {
        "fridge": {"f1": 0.5456, "nde": 0.7969},
        "microwave": {"f1": 0.6458, "nde": 0.8035},
        "dishwasher": {"f1": 0.6426, "nde": 0.6971},
        "washing_machine": {"f1": 0.0000, "nde": 0.0000},
    }
    attempt_1 = {
        "fridge": {"f1": 0.0019, "nde": 0.9998},
        "microwave": {"f1": 0.5153, "nde": 1.8660},
        "dishwasher": {"f1": 0.7108, "nde": 0.7620},
        "washing_machine": {"f1": 0.0000, "nde": 0.0000},
    }

    comparison_rows = []
    for app in config_redd.appliances:
        curr_f1 = cross.get(app, {}).get("f1", 0.0)
        curr_nde = cross.get(app, {}).get("nde", 0.0)
        b25_f1 = baseline_b25[app]["f1"]
        att1_f1 = attempt_1[app]["f1"]

        diff_vs_base = curr_f1 - b25_f1 if not np.isnan(curr_f1) else np.nan
        diff_vs_att1 = curr_f1 - att1_f1 if not np.isnan(curr_f1) else np.nan

        if np.isnan(curr_f1) or (app == "washing_machine" and b25_f1 == 0.0):
            verdict = "WASH (Unmetered / Inactive)"
        elif diff_vs_base > 0.02:
            verdict = "HELPS (Beats Baseline)"
        elif diff_vs_base < -0.05:
            verdict = "HURTS (Below Baseline)"
        else:
            verdict = "MATCHES (Near Baseline)"

        comparison_rows.append({
            "Appliance": app,
            "Baseline F1": f"{b25_f1:.4f}",
            "Attempt 1 F1": f"{att1_f1:.4f}",
            "Transfer F1": f"{curr_f1:.4f}" if not np.isnan(curr_f1) else "nan",
            "Delta vs Base": f"{diff_vs_base:+.4f}" if not np.isnan(diff_vs_base) else "nan",
            "Delta vs Att1": f"{diff_vs_att1:+.4f}" if not np.isnan(diff_vs_att1) else "nan",
            "Transfer NDE": f"{curr_nde:.4f}" if not np.isnan(curr_nde) else "nan",
            "Verdict": verdict,
        })

    comp_df = pd.DataFrame(comparison_rows)
    print("\n==========================================================================================================")
    print("                    3-WAY HEAD-TO-HEAD COMPARISON ON HELD-OUT HOUSE 2                                      ")
    print("==========================================================================================================")
    print(comp_df.to_string(index=False))
    print("==========================================================================================================\n")


if __name__ == "__main__":
    run_pretraining_and_finetuning()
