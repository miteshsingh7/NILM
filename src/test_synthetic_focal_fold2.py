"""Refined single-fold validation: Selective Synthetic Data Augmentation + Selective Focal Loss (REDD Fold 2).

Key Refinements:
1. Exclude fridge from synthetic burst-injection entirely. Fridge is masked out on synthetic windows
   (app_mask=0.0) so it trains on 100% real data only.
2. Fridge retains its original working loss: on_weight=8.0 upweighted BCE + gated MSE.
3. Binary Focal Loss (gamma=2.0, alpha=0.25) applied strictly to microwave, dishwasher, and washing machine.
4. Validation and test sets remain strictly 100% real, untouched.
"""

import json
from pathlib import Path
import pandas as pd
import torch

from src.config import NILMConfig
from src.train import prepare_datasets, train_model
from src.evaluate import run_evaluation
from src.loho_cv import print_active_sample_audit
from src.synthetic_augmentation import build_augmented_training_dataset


def run_experiment():
    fold = 2
    ckpt_dir = Path("checkpoints/refined_synth_focal_fold_2")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = ckpt_dir / "best_model.pt"
    eval_result_path = ckpt_dir / "eval_results.json"

    # Refined Configuration:
    # - on_weight=8.0 (retained for regression active-state upweighting)
    # - Fridge: original standard BCE + 8.0x weighted gated MSE
    # - Microwave, Dishwasher, Washing Machine: Focal Loss (gamma=2.0, alpha=0.25) on on/off heads
    focal_target_apps = ["microwave", "dishwasher", "washing_machine"]
    config = NILMConfig(
        epochs=35,
        batch_size=128,
        lr=1e-3,
        held_out_house=fold,
        checkpoint_dir=str(ckpt_dir),
        on_weight=8.0,
        use_focal_loss=True,
        focal_gamma=2.0,
        focal_alpha=0.25,
        focal_appliances=focal_target_apps,
    )

    print(f"\n================================================================================")
    print(f"   STARTING TEST: REFINED SYNTHETIC AUGMENTATION + SELECTIVE FOCAL LOSS (FOLD {fold})   ")
    print(f"================================================================================\n")
    print(f"Configuration:")
    print(f"  - Held-out House: House {fold}")
    print(f"  - REFINEMENT 1: Fridge excluded from synthetic bursts & masked out on synthetic windows (real data only)")
    print(f"  - REFINEMENT 2: Fridge uses original working loss (on_weight=8.0, BCE, gated MSE)")
    print(f"  - REFINEMENT 3: Selective Focal Loss (gamma={config.focal_gamma}, alpha={config.focal_alpha}) on: {focal_target_apps}")
    print(f"  - Regression Active Upweighting: on_weight={config.on_weight}x")
    print(f"  - Oversampling: boost_weight=2.5 on mixed training pool")
    print(f"  - Val & Test Sets: Strictly 100% real, natural distribution")
    print(f"  - Device: {config.device}\n")

    # 1. Dataset Preparation
    train_ds, val_ds, test_ds, norm_params, train_dfs, house_dfs = prepare_datasets(
        config=config,
        data_dir="data/processed",
        redd_dir="data/raw/redd",
        return_raw_splits=True,
    )

    # 2. Print Active Sample Audit
    print_active_sample_audit(house_dfs, config, fold=fold)

    # 3. Synthesize Kelly & Knottenbelt Augmentation Windows (EXCLUDING FRIDGE)
    augmented_train_ds, audit_info = build_augmented_training_dataset(
        real_train_dataset=train_ds,
        train_dfs=train_dfs,
        appliances=config.appliances,
        thresholds=config.thresholds,
        norm_params=norm_params,
        synthetic_ratio=1.0,  # 1:1 mix
        window_length=config.window_length,
        exclude_appliances=["fridge"],  # Fridge trains on real data only!
        random_seed=42,
    )

    # 4. Train Model with boost_weight=2.5 and Selective Focal Loss
    model, history = train_model(
        config=config,
        train_dataset=augmented_train_ds,
        val_dataset=val_ds,
        norm_params=norm_params,
        checkpoint_dir=str(ckpt_dir),
        use_oversampling=True,
        boost_weight=2.5,
        resume=False,
    )

    # 5. Evaluation on Unseen House 2
    print(f"\n================ EXPLICIT EVALUATION FOR FOLD {fold} ================\n")
    eval_results, _ = run_evaluation(
        checkpoint_path=str(best_model_path),
        data_dir="data/processed",
        redd_dir="data/raw/redd",
        dataset_type="redd",
        output_path=str(eval_result_path),
        device=config.device,
        held_out_house=fold,
    )

    # 6. Print 3-Way Direct Comparison Table
    with open(eval_result_path, "r") as f:
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
    for app in config.appliances:
        curr_f1 = cross.get(app, {}).get("f1", 0.0)
        curr_nde = cross.get(app, {}).get("nde", 0.0)
        b25_f1 = baseline_b25[app]["f1"]
        b25_nde = baseline_b25[app]["nde"]
        att1_f1 = attempt_1[app]["f1"]

        diff_vs_baseline = curr_f1 - b25_f1
        diff_vs_att1 = curr_f1 - att1_f1

        if diff_vs_baseline > 0.02:
            verdict = "HELPS (Beats Baseline)"
        elif diff_vs_baseline < -0.02:
            verdict = "HURTS (Below Baseline)"
        else:
            verdict = "WASH (Matches Baseline)"

        comparison_rows.append({
            "Appliance": app,
            "Baseline (Boost=2.5) F1": f"{b25_f1:.4f}",
            "Attempt 1 (Blanket) F1": f"{att1_f1:.4f}",
            "Refined (Current) F1": f"{curr_f1:.4f}",
            "Delta vs Baseline": f"{diff_vs_baseline:+.4f}",
            "Delta vs Attempt 1": f"{diff_vs_att1:+.4f}",
            "Current NDE": f"{curr_nde:.4f}",
            "Verdict": verdict,
        })

    print("\n==========================================================================================================")
    print("                    3-WAY HEAD-TO-HEAD COMPARISON ON HELD-OUT HOUSE 2                                      ")
    print("==========================================================================================================")
    comp_df = pd.DataFrame(comparison_rows)
    print(comp_df.to_string(index=False))
    print("==========================================================================================================\n")


if __name__ == "__main__":
    run_experiment()
