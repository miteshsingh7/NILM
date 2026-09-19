"""Retrain Folds 5 & 6 with gradient clipping + NaN guards, and run complete 6-fold evaluation.

Folds 1-4 are verified clean (0/76 NaN tensors) and reused as-is.
Folds 5 and 6 are retrained from scratch (seed 42) with grad_clip_norm=2.0 and strict parameter guards.
Device assertion checks ensure model and tensors match exactly across all evaluation paths.
"""

import hashlib
import json
import os
import random
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEFAULT_APPLIANCES, NILMConfig
from src.data_pipeline import (
    NILMDataset,
    create_sliding_windows,
)
from src.diagnostics import compute_phase0_diagnostics
from src.evaluate import evaluate_dataset, run_evaluation
from src.model import DecoupledTemporalNILM
from src.train import train_model, prepare_datasets
from src.utils import compute_f1_score


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def verify_checkpoint_tensors(ckpt_path: Path) -> int:
    """Verifies that checkpoint contains 0 NaN or Inf parameter tensors."""
    assert ckpt_path.exists(), f"Checkpoint not found at {ckpt_path}"
    state = torch.load(str(ckpt_path), map_location="cpu")
    weights = state.get("model_state_dict", state)
    nan_count = sum(1 for name, p in weights.items() if torch.isnan(p).any() or torch.isinf(p).any())
    return nan_count


def main():
    set_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Executing on device: {device}")

    checkpoint_base = Path("checkpoints/loho_cv_combined_phase7")
    folds_to_retrain = [5, 6]
    all_folds = [1, 2, 3, 4, 5, 6]

    print("\n==========================================================================================")
    print("      STEP 1: VERIFY EXISTING CHECKPOINTS FOR FOLDS 1-4 (PRE-RETRAINING AUDIT)           ")
    print("==========================================================================================")
    for f in [1, 2, 3, 4]:
        ckpt = checkpoint_base / f"fold_{f}" / "best_model.pt"
        nan_cnt = verify_checkpoint_tensors(ckpt)
        print(f"Fold {f} ({ckpt}): {nan_cnt}/76 NaN/Inf tensors -> {'VERIFIED CLEAN' if nan_cnt == 0 else 'CORRUPTED'}")
        assert nan_cnt == 0, f"Fold {f} checkpoint has corrupted tensors!"

    print("\n==========================================================================================")
    print("      STEP 2: RETRAIN FOLDS 5 & 6 WITH GRADIENT CLIPPING AND NAN GUARDS                  ")
    print("==========================================================================================")
    for fold in folds_to_retrain:
        t_fold_start = time.time()
        ckpt_dir = checkpoint_base / f"fold_{fold}"
        print(f"\n{'#' * 90}")
        print(f"   STARTING CLEAN RETRAINING: FOLD {fold}/6 (HELD-OUT HOUSE {fold})")
        print(f"   Destination: {ckpt_dir}")
        print(f"{'#' * 90}\n")

        # Clean old corrupted state files to ensure a fresh, non-resumed training run
        for fname in ["best_model.pt", "latest_checkpoint.pt", "history.json"]:
            old_f = ckpt_dir / fname
            if old_f.exists():
                old_f.unlink()

        config = NILMConfig(
            epochs=35,
            batch_size=128,
            lr=1e-3,
            held_out_house=fold,
            checkpoint_dir=str(ckpt_dir),
            norm_type="groupnorm",
            model_type="decoupled_temporal",
            lstm_hidden_size=48,
            in_channels=1,
            predict_transitions=False,
            num_groups=8,
            grad_clip_norm=2.0,
            device=str(device),
        )

        train_ds, val_ds, test_ds, norm_params = prepare_datasets(
            config=config,
            data_dir="data/processed",
            redd_dir="data/raw/redd",
        )

        model, history = train_model(
            config=config,
            train_dataset=train_ds,
            val_dataset=val_ds,
            norm_params=norm_params,
            checkpoint_dir=str(ckpt_dir),
            use_oversampling=True,
            boost_weight=2.5,
            resume=False,
        )

        best_ckpt = ckpt_dir / "best_model.pt"
        nan_cnt = verify_checkpoint_tensors(best_ckpt)
        print(f"\n[Verification] Fold {fold} retrained best_model.pt: {nan_cnt}/76 NaN/Inf tensors.")
        assert nan_cnt == 0, f"FATAL: Fold {fold} retrained checkpoint still has NaN weights!"
        print(f"✅ Fold {fold} retraining SUCCESSFUL with 0/76 NaN tensors (Elapsed: {(time.time() - t_fold_start)/60:.2f} min)")

    print("\n==========================================================================================")
    print("      STEP 3: RUN COMPLETE 6-FOLD EVALUATION ON ALL THREE THRESHOLD PROTOCOLS             ")
    print("==========================================================================================")

    fold_results = {}
    fold_p10_results = {}
    fold_shrinkage_results = {}

    # Pre-compute source P10 baseload values for Zero-Touch Adaptation
    p10_all_houses = {}
    for h in all_folds:
        df_h = pd.read_csv(f"data/processed/redd_real_house_{h}.csv", index_col=0, parse_dates=True)
        p10_all_houses[h] = float(np.percentile(df_h["mains"].dropna(), 10))

    for fold in all_folds:
        print(f"\n{'=' * 85}")
        print(f"EVALUATING FOLD {fold}/6 (HELD-OUT HOUSE {fold})")
        print(f"{'=' * 85}")
        ckpt_dir = checkpoint_base / f"fold_{fold}"
        best_ckpt_path = ckpt_dir / "best_model.pt"
        eval_output_path = ckpt_dir / "eval_results.json"

        # 1. Standard full-house evaluation (Static threshold = 0.50 & full diagnostics)
        eval_res, comp_df = run_evaluation(
            checkpoint_path=str(best_ckpt_path),
            data_dir="data/processed",
            redd_dir="data/raw/redd",
            output_path=str(eval_output_path),
            device=str(device),
            held_out_house=fold,
        )
        fold_results[fold] = eval_res

        # 2. Protocol A: Zero-Touch P10 Unsupervised Adaptation (Refrigerator Only)
        train_houses = [h for h in all_folds if h != fold]
        mean_p10_source = float(np.mean([p10_all_houses[th] for th in train_houses]))
        p10_target = p10_all_houses[fold]
        beta_p10 = 0.0012
        theta_p10 = float(np.clip(0.50 + beta_p10 * (p10_target - mean_p10_source), 0.20, 0.80))

        # Dataset loader for test house
        config_eval = NILMConfig(
            held_out_house=fold,
            checkpoint_dir=str(ckpt_dir),
            norm_type="groupnorm",
            model_type="decoupled_temporal",
            lstm_hidden_size=48,
            num_groups=8,
            device=str(device),
        )
        _, _, test_ds, norm_params = prepare_datasets(
            config=config_eval,
            data_dir="data/processed",
            redd_dir="data/raw/redd",
        )

        raw_model = DecoupledTemporalNILM(
            appliances=DEFAULT_APPLIANCES,
            in_channels=1,
            lstm_hidden=48,
            norm_type="groupnorm",
            num_groups=8,
        ).to(device)

        ckpt_data = torch.load(best_ckpt_path, map_location=device)
        clean_st = {k[7:] if k.startswith("module.") else k: v for k, v in ckpt_data["model_state_dict"].items()}
        raw_model.load_state_dict(clean_st)
        raw_model.eval()

        loader = DataLoader(test_ds, batch_size=128, shuffle=False)
        all_o_pred, all_o_true, all_mask = [], [], []
        with torch.no_grad():
            for b in loader:
                xb = b[0].to(device)
                assert next(raw_model.parameters()).device == xb.device, "Device mismatch between model and inputs!"
                yp = b[1]
                yo = b[2]
                mk = b[3]
                out = raw_model(xb)
                all_o_pred.append(out["on_off"].cpu().numpy())
                all_o_true.append(yo.numpy())
                all_mask.append(mk.numpy())

        preds_onoff = np.concatenate(all_o_pred, axis=0)
        trues_onoff = np.concatenate(all_o_true, axis=0)
        masks_app = np.concatenate(all_mask, axis=0)

        # Fridge is index 0
        fridge_metrics = eval_res["cross_household"].get("fridge", {})
        oracle_f1_fridge = fridge_metrics.get("oracle_f1", np.nan)
        f_mask = masks_app[..., 0] > 0.5

        if np.sum(f_mask) > 0 and not np.isnan(oracle_f1_fridge):
            f_pred = preds_onoff[..., 0][f_mask]
            f_true = trues_onoff[..., 0][f_mask]
            clf_p10 = compute_f1_score(f_true, f_pred, threshold=theta_p10)
            p10_f1 = clf_p10["f1"]
            rec_pct = (p10_f1 / oracle_f1_fridge * 100) if oracle_f1_fridge > 0 else 0.0
        else:
            p10_f1 = np.nan
            rec_pct = np.nan

        print(f"\n--- [Fold {fold}] Zero-Touch P10 Adaptation (Fridge) ---")
        print(f"  Target P10: {p10_target:.1f} W (Source Mean: {mean_p10_source:.1f} W)")
        print(f"  Unsupervised Adapted Threshold: {theta_p10:.4f}")
        print(f"  Baseline F1 (@0.50)           : {fridge_metrics.get('f1', np.nan):.4f}")
        print(f"  Zero-Touch P10 F1             : {p10_f1:.4f}")
        print(f"  Oracle Ceiling F1             : {oracle_f1_fridge:.4f}")
        print(f"  Oracle Ceiling Recovered      : {rec_pct:.2f}%")

        fold_p10_results[fold] = {
            "p10_target": p10_target,
            "mean_p10_source": mean_p10_source,
            "theta_p10": theta_p10,
            "f1_at_50": fridge_metrics.get("f1", np.nan),
            "f1_p10": p10_f1,
            "oracle_f1": oracle_f1_fridge,
            "recovered_pct": rec_pct,
        }

        # 3. Protocol B: Commissioning James-Stein Shrinkage Calibration (All 4 Appliances)
        print(f"\n--- [Fold {fold}] Commissioning James-Stein Shrinkage Calibration ---")
        df_target = pd.read_csv(f"data/processed/redd_real_house_{fold}.csv", index_col=0, parse_dates=True)
        calib_stride = 149
        eval_stride = 599
        window_length = 599
        day1_samples = 24 * 3600 // 6  # 14,400 samples

        df_calib = df_target.iloc[:day1_samples]
        df_eval = df_target.iloc[day1_samples:]

        calib_x, calib_yp, calib_yo, calib_m = create_sliding_windows(
            df_calib, DEFAULT_APPLIANCES, norm_params, window_length=window_length, stride=calib_stride
        )
        eval_x, eval_yp, eval_yo, eval_m = create_sliding_windows(
            df_eval, DEFAULT_APPLIANCES, norm_params, window_length=window_length, stride=eval_stride
        )

        print(f"  Calibration Windows [0:24h]     : {len(calib_x)}")
        print(f"  Held-Out Eval Windows [24h:end] : {len(eval_x)} (Strict Non-Overlapping Boundary Verified)")

        shrinkage_metrics = {}
        if len(calib_x) > 0 and len(eval_x) > 0:
            calib_ds = NILMDataset(calib_x, calib_yp, calib_yo, calib_m)
            eval_ds = NILMDataset(eval_x, eval_yp, eval_yo, eval_m)

            c_loader = DataLoader(calib_ds, batch_size=128, shuffle=False)
            e_loader = DataLoader(eval_ds, batch_size=128, shuffle=False)

            c_preds_list, c_trues_list, c_masks_list = [], [], []
            with torch.no_grad():
                for b in c_loader:
                    xb = b[0].to(device)
                    assert next(raw_model.parameters()).device == xb.device, "Device mismatch!"
                    out = raw_model(xb)
                    c_preds_list.append(out["on_off"].cpu().numpy())
                    c_trues_list.append(b[2].numpy())
                    c_masks_list.append(b[3].numpy())

            cp = np.concatenate(c_preds_list, axis=0)
            ct = np.concatenate(c_trues_list, axis=0)
            cm = np.concatenate(c_masks_list, axis=0)

            e_preds_list, e_trues_list, e_masks_list = [], [], []
            with torch.no_grad():
                for b in e_loader:
                    xb = b[0].to(device)
                    assert next(raw_model.parameters()).device == xb.device, "Device mismatch!"
                    out = raw_model(xb)
                    e_preds_list.append(out["on_off"].cpu().numpy())
                    e_trues_list.append(b[2].numpy())
                    e_masks_list.append(b[3].numpy())

            ep = np.concatenate(e_preds_list, axis=0)
            et = np.concatenate(e_trues_list, axis=0)
            em = np.concatenate(e_masks_list, axis=0)

            for i, app in enumerate(DEFAULT_APPLIANCES):
                app_mask_eval = em[..., i] > 0.5
                if np.sum(app_mask_eval) == 0:
                    shrinkage_metrics[app] = {"f1": np.nan, "oracle_f1": np.nan, "theta_shrink": np.nan, "n_active_calib": 0, "alpha": 0.0}
                    continue

                app_mask_calib = cm[..., i] > 0.5
                valid_calib_t = ct[..., i][app_mask_calib]
                valid_calib_p = cp[..., i][app_mask_calib]
                n_active_calib = int(np.sum(valid_calib_t > 0.5))

                theta_source = 0.50
                if n_active_calib > 0:
                    best_th = 0.50
                    best_f1 = -1.0
                    for th_cand in np.linspace(0.1, 0.9, 81):
                        cand_f1 = compute_f1_score(valid_calib_t, valid_calib_p, threshold=th_cand)["f1"]
                        if cand_f1 > best_f1:
                            best_f1 = cand_f1
                            best_th = th_cand
                    theta_target = best_th
                else:
                    theta_target = theta_source

                alpha = float(n_active_calib / (n_active_calib + 50))
                theta_shrink = float(alpha * theta_target + (1.0 - alpha) * theta_source)

                valid_eval_t = et[..., i][app_mask_eval]
                valid_eval_p = ep[..., i][app_mask_eval]

                if np.sum(valid_eval_t > 0.5) > 0:
                    res_eval = compute_f1_score(valid_eval_t, valid_eval_p, threshold=theta_shrink)
                    diag_eval = compute_phase0_diagnostics(valid_eval_t, valid_eval_p)
                    shrinkage_metrics[app] = {
                        "f1": res_eval["f1"],
                        "precision": res_eval["precision"],
                        "recall": res_eval["recall"],
                        "oracle_f1": diag_eval["oracle_f1"],
                        "oracle_th": diag_eval["oracle_threshold"],
                        "theta_shrink": theta_shrink,
                        "n_active_calib": n_active_calib,
                        "alpha": alpha,
                    }
                    print(f"  * {app:15s}: Calib N_act={n_active_calib:4d} (alpha={alpha:.2f}) -> Theta={theta_shrink:.4f} -> Eval F1={res_eval['f1']:.4f} (Oracle: {diag_eval['oracle_f1']:.4f})")
                else:
                    shrinkage_metrics[app] = {"f1": np.nan, "oracle_f1": np.nan, "theta_shrink": theta_shrink, "n_active_calib": n_active_calib, "alpha": alpha}
        else:
            for app in DEFAULT_APPLIANCES:
                shrinkage_metrics[app] = {"f1": np.nan, "oracle_f1": np.nan, "theta_shrink": 0.5, "n_active_calib": 0, "alpha": 0.0}

        fold_shrinkage_results[fold] = shrinkage_metrics

    # ------------------------------------------------------------------------------
    # 4. Assembling and Printing Grand & Sanitized Comparison Tables
    # ------------------------------------------------------------------------------
    full_summary = {
        "fold_results": fold_results,
        "p10_results": fold_p10_results,
        "shrinkage_results": fold_shrinkage_results,
    }
    with open(checkpoint_base / "retrained_phase7_full_results.json", "w") as fp:
        json.dump(full_summary, fp, indent=2)

    print("\n" + "=" * 115)
    print("             GRAND COMBINED ARCHITECTURAL BENCHMARK (ALL 6 FOLDS) - RETRAINED")
    print("=" * 115)
    print(f"{'Fold':7s} | {'Fridge F1':10s} {'Fridge AP':10s} {'Fridge NDE':10s} | {'Micr F1':10s} {'Micr AP':10s} {'Micr NDE':10s} | {'Dish F1':10s} {'Dish AP':10s} {'Dish NDE':10s} | {'Wash F1':10s} {'Wash AP':10s} {'Wash NDE':10s}")
    print("-" * 115)

    app_pools = {app: {"f1": [], "ap": [], "nde": []} for app in DEFAULT_APPLIANCES}
    sanitized_pools = {app: {"f1": [], "ap": [], "nde": []} for app in DEFAULT_APPLIANCES}
    metered_houses = {
        "fridge": [1, 2, 3, 5, 6],
        "microwave": [1, 2, 3],
        "dishwasher": [1, 2, 3, 4],
        "washing_machine": [1, 3, 4],
    }

    for fold in all_folds:
        cr = fold_results[fold]["cross_household"]
        row_str = f"Fold {fold} |"
        for app in DEFAULT_APPLIANCES:
            m = cr.get(app, {})
            f1 = m.get("f1", np.nan)
            ap = m.get("ap", np.nan)
            nde = m.get("nde", np.nan)
            f1_s = f"{f1:10.4f}" if not np.isnan(f1) else "       N/A"
            ap_s = f"{ap:10.4f}" if not np.isnan(ap) else "       N/A"
            nde_s = f"{nde:10.4f}" if not np.isnan(nde) else "       N/A"
            row_str += f" {f1_s} {ap_s} {nde_s} |"

            if not np.isnan(f1):
                app_pools[app]["f1"].append(f1)
                app_pools[app]["ap"].append(ap)
                app_pools[app]["nde"].append(nde)

            if fold in metered_houses[app] and not np.isnan(f1):
                sanitized_pools[app]["f1"].append(f1)
                sanitized_pools[app]["ap"].append(ap)
                sanitized_pools[app]["nde"].append(nde)

        print(row_str)

    print("-" * 115)
    mean_str = "Mean   |"
    for app in DEFAULT_APPLIANCES:
        f1_m = np.mean(app_pools[app]["f1"]) if app_pools[app]["f1"] else np.nan
        ap_m = np.mean(app_pools[app]["ap"]) if app_pools[app]["ap"] else np.nan
        nde_m = np.mean(app_pools[app]["nde"]) if app_pools[app]["nde"] else np.nan
        mean_str += f" {f1_m:10.4f} {ap_m:10.4f} {nde_m:10.4f} |"
    print(mean_str)
    print("=" * 115)

    print("\n" + "=" * 105)
    print("      SANITIZED COMBINED BENCHMARK (VALID METERED ACTIVE TEST HOUSES ONLY)")
    print("=" * 105)
    print(f"{'Appliance':16s} | {'Sanitized F1':14s} {'Sanitized AP':14s} {'Sanitized NDE':14s} | {'Active Test Houses'}")
    print("-" * 105)
    for app in DEFAULT_APPLIANCES:
        f1_s = np.mean(sanitized_pools[app]["f1"]) if sanitized_pools[app]["f1"] else np.nan
        ap_s = np.mean(sanitized_pools[app]["ap"]) if sanitized_pools[app]["ap"] else np.nan
        nde_s = np.mean(sanitized_pools[app]["nde"]) if sanitized_pools[app]["nde"] else np.nan
        active_str = ", ".join([f"House {h}" for h in metered_houses[app]])
        print(f"{app:16s} | {f1_s:14.4f} {ap_s:14.4f} {nde_s:14.4f} | {active_str}")
    print("=" * 105)

    print("\n" + "=" * 90)
    print("SECTION 5: CALIBRATION PROTOCOLS EVALUATION")
    print("=" * 90)

    print("\n--- Protocol A: Zero-Touch P10 Unsupervised Adaptation (Refrigerator Only) ---")
    print(f"{'Fold':7s} | {'Target P10 (W)':14s} {'Adapted Theta':15s} {'F1 (@0.50)':12s} {'P10 F1':12s} {'Oracle F1':12s} {'% Recovered':12s}")
    print("-" * 90)
    p10_f1_list, oracle_f1_list, f1_50_list = [], [], []
    for fold in all_folds:
        p = fold_p10_results[fold]
        t_p10 = p["p10_target"]
        th_p10 = p["theta_p10"]
        f50 = p["f1_at_50"]
        fp10 = p["f1_p10"]
        orc = p["oracle_f1"]
        rec = p["recovered_pct"]

        f50_s = f"{f50:12.4f}" if not np.isnan(f50) else "         nan"
        fp10_s = f"{fp10:12.4f}" if not np.isnan(fp10) else "         N/A"
        orc_s = f"{orc:12.4f}" if not np.isnan(orc) else "         N/A"
        rec_s = f"{rec:11.2f}%" if not np.isnan(rec) else "         N/A"

        print(f"Fold {fold:2d} | {t_p10:14.1f} {th_p10:15.4f} {f50_s} {fp10_s} {orc_s} {rec_s}")
        if not np.isnan(fp10):
            p10_f1_list.append(fp10)
        if not np.isnan(orc):
            oracle_f1_list.append(orc)
        if not np.isnan(f50):
            f1_50_list.append(f50)

    print("-" * 90)
    mean_f50 = np.mean(f1_50_list) if f1_50_list else np.nan
    mean_fp10 = np.mean(p10_f1_list) if p10_f1_list else np.nan
    mean_orc = np.mean(oracle_f1_list) if oracle_f1_list else np.nan
    mean_rec = (mean_fp10 / mean_orc * 100) if mean_orc > 0 else np.nan
    print(f"Mean   | {'N/A':14s} {'N/A':15s} {mean_f50:12.4f} {mean_fp10:12.4f} {mean_orc:12.4f} {mean_rec:11.2f}%")

    print("\n--- Protocol B: Commissioning James-Stein Shrinkage Calibration (All 4 Appliances) ---")
    print("Evaluated strictly on held-out post-24h evaluation suffix [24h:end]")
    print(f"{'Appliance':16s} | {'Shrinkage F1':14s} {'Oracle F1':14s} {'% of Ceiling':14s} | {'v3 Same-House Base':18s} {'Improvement':14s}")
    print("-" * 90)

    v3_baselines = {
        "fridge": 0.4697,  # 24h calibrated baseline across evaluable houses [1, 2, 3, 5, 6]
        "microwave": 0.5315,  # Same-house v3 baseline across evaluable houses [1, 2, 3]
        "dishwasher": 0.3625,  # 24h calibrated baseline across evaluable houses [1, 2, 3, 4]
        "washing_machine": 0.3112,  # Same-house v3 baseline across evaluable houses [1, 3, 4]
    }

    for app in DEFAULT_APPLIANCES:
        sh_f1s = []
        orc_f1s = []
        for fold in all_folds:
            if fold in metered_houses[app]:
                m = fold_shrinkage_results[fold].get(app, {})
                f1_val = m.get("f1", np.nan)
                orc_val = m.get("oracle_f1", np.nan)
                if not np.isnan(f1_val):
                    sh_f1s.append(f1_val)
                if not np.isnan(orc_val):
                    orc_f1s.append(orc_val)

        mean_sh = np.mean(sh_f1s) if sh_f1s else np.nan
        mean_orc = np.mean(orc_f1s) if orc_f1s else np.nan
        pct_ceil = (mean_sh / mean_orc * 100) if mean_orc > 0 else np.nan
        v3_b = v3_baselines.get(app, np.nan)
        diff = mean_sh - v3_b

        diff_str = f"{diff:+.4f}" if not np.isnan(diff) else "N/A"
        flag_str = " (FLAG: NOT_VALIDATED)" if diff < 0 else " (VERIFIED_GAIN)"
        print(f"{app:16s} | {mean_sh:14.4f} {mean_orc:14.4f} {pct_ceil:13.2f}% | {v3_b:18.4f} {diff_str + flag_str}")
    print("=" * 90)

    print("\n==========================================================================================")
    print("      STEP 4: DIRECT BENCHMARK COMPARISON VS V3 BASELINE (0.4697) & PHASE 7 (0.4810)      ")
    print("==========================================================================================")
    print(f"v3 Baseline (BatchNorm, 24h calibration, 5-fold mean)        : 0.4697")
    print(f"Phase 7 Claim (Isolated Shrinkage, 5-fold mean)              : 0.4810")
    print(f"Retrained Combined Architecture - Zero-Touch P10 (Fridge)    : {mean_fp10:.4f}")
    fridge_sh_f1 = np.mean([fold_shrinkage_results[fold]['fridge']['f1'] for fold in metered_houses['fridge'] if not np.isnan(fold_shrinkage_results[fold]['fridge']['f1'])])
    print(f"Retrained Combined Architecture - Shrinkage Eval (Fridge)    : {fridge_sh_f1:.4f}")


if __name__ == "__main__":
    main()
