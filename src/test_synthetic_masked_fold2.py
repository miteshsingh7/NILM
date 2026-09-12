"""Single-fold validation: Kelly & Knottenbelt Synthetic Data Augmentation (Micr/Dish/Wash only)
with Strict Per-Sample Masking & Verified Baseline Loss (Uniform on_weight=8.0 BCE).

Requirements:
1. Loss: Uniform on_weight=8.0 BCE across all 4 heads, gated MSE, boost_weight=2.5 oversampling.
   NO focal loss.
2. Synthetic Augmentation: Active snippets injected for microwave, dishwasher, washing machine only.
   Fridge is strictly excluded from synthetic windows and trains on real data only.
3. Strict Per-Sample Masking: appliance_mask zeros out fridge loss per-sample on synthetic windows.
   Print explicit check confirming synthetic fridge loss contribution is strictly zero.
4. Full 35 Epochs: early_stopping_patience=35 to run all 35 epochs without early termination.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import json
import numpy as np
import pandas as pd
import torch

from src.config import NILMConfig
from src.loss import MultiApplianceLoss
from src.train import prepare_datasets, train_model
from src.evaluate import run_evaluation
from src.loho_cv import print_active_sample_audit
from src.synthetic_augmentation import build_augmented_training_dataset


def verify_per_sample_masking_explicit(config: NILMConfig):
    """Explicitly verifies that synthetic windows contribute strictly 0.0 to fridge loss & gradients."""
    print("\n================================================================================")
    print("      EXPLICIT VERIFICATION: PER-SAMPLE MASKING ON SYNTHETIC WINDOWS           ")
    print("================================================================================")
    loss_fn = MultiApplianceLoss(
        appliances=config.appliances,
        lambda_bce=config.lambda_loss,
        on_weight=config.on_weight,
        gated=True,
        use_focal_loss=False,
    )
    batch_size = 4
    length = 599
    num_apps = len(config.appliances)

    # 2 real windows (fridge present: mask=1.0) + 2 synthetic windows (fridge excluded: mask=0.0)
    app_mask = torch.tensor([
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 1.0],
    ], dtype=torch.float32)

    torch.manual_seed(42)
    p_pred = torch.randn(batch_size, length, num_apps, requires_grad=True)
    p_true = torch.rand(batch_size, length, num_apps)
    o_pred = torch.rand(batch_size, length, num_apps, requires_grad=True)
    o_true = (torch.rand(batch_size, length, num_apps) > 0.7).float()

    loss_1, bd_1 = loss_fn(p_pred, p_true, o_pred, o_true, appliance_mask=app_mask)
    loss_1.backward()

    synth_p_grad = p_pred.grad[2:, :, 0].abs().max().item()
    synth_o_grad = o_pred.grad[2:, :, 0].abs().max().item()
    real_p_grad = p_pred.grad[:2, :, 0].abs().max().item()
    real_o_grad = o_pred.grad[:2, :, 0].abs().max().item()

    # Perturb synthetic samples by 1,000,000 to verify invariance
    p_pred_2 = p_pred.detach().clone()
    o_pred_2 = o_pred.detach().clone()
    p_pred_2[2:, :, 0] = 1_000_000.0
    o_pred_2[2:, :, 0] = 0.999999
    loss_2, bd_2 = loss_fn(p_pred_2, p_true, o_pred_2, o_true, appliance_mask=app_mask)

    fridge_loss_diff = abs(bd_1["fridge_loss"] - bd_2["fridge_loss"])
    total_loss_diff = abs(loss_1.item() - loss_2.item())

    print(f"Mask configuration: Sample 0,1 (Real, mask=1.0) | Sample 2,3 (Synthetic, mask=0.0)")
    print(f"Synthetic Fridge Power Gradient: max|grad| = {synth_p_grad:.10f}")
    print(f"Synthetic Fridge OnOff Gradient: max|grad| = {synth_o_grad:.10f}")
    print(f"Real Fridge Power Gradient:      max|grad| = {real_p_grad:.10f}")
    print(f"Real Fridge OnOff Gradient:      max|grad| = {real_o_grad:.10f}")
    print(f"Perturbation Check (1,000,000x on synthetic fridge):")
    print(f"  Fridge Loss Difference: {fridge_loss_diff:.12f}")
    print(f"  Total Loss Difference:  {total_loss_diff:.12f}")

    assert synth_p_grad == 0.0, "FATAL: Synthetic fridge power gradient is NOT zero!"
    assert synth_o_grad == 0.0, "FATAL: Synthetic fridge onoff gradient is NOT zero!"
    assert fridge_loss_diff == 0.0, "FATAL: Fridge loss changed after perturbing synthetic windows!"
    assert total_loss_diff == 0.0, "FATAL: Total loss changed after perturbing synthetic windows!"
    print(">>> CHECK PASSED: Fridge loss contribution from synthetic windows is STRICTLY ZERO (0.0).")
    print("================================================================================\n")


def run_experiment():
    fold = 2
    ckpt_dir = Path("checkpoints/synthetic_masked_fold_2")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = ckpt_dir / "best_model.pt"
    eval_result_path = ckpt_dir / "eval_results.json"

    # Strict Configuration:
    # - Uniform on_weight=8.0 across all 4 heads
    # - Standard BCE (use_focal_loss=False) across all heads
    # - Gated MSE
    # - boost_weight=2.5 oversampling
    # - early_stopping_patience=35 (run full 35 epochs)
    config = NILMConfig(
        epochs=35,
        early_stopping_patience=35,
        batch_size=128,
        lr=1e-3,
        held_out_house=fold,
        checkpoint_dir=str(ckpt_dir),
        on_weight=8.0,
        lambda_loss=1.0,
        use_focal_loss=False,
    )

    print(f"\n================================================================================")
    print(f"   STARTING TEST: SYNTHETIC AUGMENTATION + STRICT MASKING + UNIFORM BCE (FOLD {fold})   ")
    print(f"================================================================================\n")
    print(f"Configuration:")
    print(f"  - Held-out House: House {fold}")
    print(f"  - Synthetic Augmentation: Enabled for ['microwave', 'dishwasher', 'washing_machine']")
    print(f"  - Fridge Augmentation: STRICTLY EXCLUDED (real data only)")
    print(f"  - Loss Formulation: Uniform on_weight={config.on_weight}x BCE across all heads (NO focal loss)")
    print(f"  - Regression Gating: Enabled (power_pred * onoff_pred)")
    print(f"  - Oversampling: boost_weight=2.5 on mixed training pool")
    print(f"  - Early Stopping Patience: {config.early_stopping_patience} (Full {config.epochs} epochs)")
    print(f"  - Val & Test Sets: Strictly 100% real, natural distribution")
    print(f"  - Device: {config.device}\n")

    # Step 0: Explicit verification of per-sample masking
    verify_per_sample_masking_explicit(config)

    # Step 1: Dataset Preparation
    train_ds, val_ds, test_ds, norm_params, train_dfs, house_dfs = prepare_datasets(
        config=config,
        data_dir="data/processed",
        redd_dir="data/raw/redd",
        return_raw_splits=True,
    )

    # Step 2: Print Active Sample Audit
    print_active_sample_audit(house_dfs, config, fold=fold)

    # Step 3: Synthesize Kelly & Knottenbelt Augmentation Windows (EXCLUDING FRIDGE)
    augmented_train_ds, audit_info = build_augmented_training_dataset(
        real_train_dataset=train_ds,
        train_dfs=train_dfs,
        appliances=config.appliances,
        thresholds=config.thresholds,
        norm_params=norm_params,
        synthetic_ratio=1.0,
        window_length=config.window_length,
        exclude_appliances=["fridge"],
        random_seed=42,
    )

    # Step 4: Train Model for Full 35 Epochs
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

    # Step 5: Evaluation on Unseen House 2
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

    # Step 6: Print Head-to-Head Comparison Table
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
            "Current F1": f"{curr_f1:.4f}" if not np.isnan(curr_f1) else "nan",
            "Delta vs Base": f"{diff_vs_base:+.4f}" if not np.isnan(diff_vs_base) else "nan",
            "Delta vs Att1": f"{diff_vs_att1:+.4f}" if not np.isnan(diff_vs_att1) else "nan",
            "Current NDE": f"{curr_nde:.4f}" if not np.isnan(curr_nde) else "nan",
            "Verdict": verdict,
        })

    comp_df = pd.DataFrame(comparison_rows)
    print("\n==========================================================================================================")
    print("                    3-WAY HEAD-TO-HEAD COMPARISON ON HELD-OUT HOUSE 2                                      ")
    print("==========================================================================================================")
    print(comp_df.to_string(index=False))
    print("==========================================================================================================\n")


if __name__ == "__main__":
    run_experiment()
