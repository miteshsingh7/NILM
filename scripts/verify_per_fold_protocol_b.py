"""Per-Fold Protocol B Verification Script across All Appliances."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    summary_path = Path("checkpoints/loho_cv_combined_phase7/loho_combined_phase7_summary.json")
    with open(summary_path) as f:
        summary_data = json.load(f)

    shrink = summary_data["protocol_b_commissioning_shrinkage"]
    day1_samples = 14400  # 24h at 6s

    appliances = ["fridge", "microwave", "dishwasher", "washing_machine"]
    metered_houses = {
        "fridge": [1, 2, 3, 5, 6],
        "microwave": [1, 2, 3],
        "dishwasher": [1, 2, 3, 4],
        "washing_machine": [1, 3, 4],
    }
    v3_baselines = {
        "fridge": 0.4697,
        "microwave": 0.3986,
        "dishwasher": 0.3625,
        "washing_machine": 0.2334,
    }

    import sys
    sys.path.insert(0, ".")
    from src.config import DEFAULT_APPLIANCE_THRESHOLDS

    # Load real sample counts from raw data CSVs using canonical thresholds
    raw_counts = {}
    for h in range(1, 7):
        fpath = Path(f"data/processed/redd_real_house_{h}.csv")
        df = pd.read_csv(fpath, index_col=0, parse_dates=True)
        raw_counts[h] = {"total_len": len(df)}
        df_cal = df.iloc[:day1_samples]
        df_eval = df.iloc[day1_samples:]
        for app in appliances:
            th = DEFAULT_APPLIANCE_THRESHOLDS[app]
            if app in df.columns:
                tot_act = int(np.sum(df[app].dropna() >= th))
                cal_act = int(np.sum(df_cal[app].dropna() >= th))
                eval_act = int(np.sum(df_eval[app].dropna() >= th))
                raw_counts[h][app] = {"tot_act": tot_act, "cal_act": cal_act, "eval_act": eval_act}
            else:
                raw_counts[h][app] = {"tot_act": 0, "cal_act": 0, "eval_act": 0}

    print("=" * 130)
    print("      PROTOCOL B (COMMISSIONING JAMES-STEIN SHRINKAGE): PER-FOLD EVIDENCE AUDIT")
    print("=" * 130)

    for app in appliances:
        print(f"\n##############################################################################################################")
        print(f"                                   APPLIANCE: {app.upper()}")
        print(f"##############################################################################################################")
        print(f"{'Fold':<6} | {'Test House':<12} | {'Calib N_act':<12} | {'Alpha':<8} | {'Calib Theta':<12} | {'Eval F1':<10} {'Eval Prec':<10} {'Eval Rec':<10} | {'Oracle F1':<10} {'(Oracle Th)':<12} | {'Eval N_act':<10} | {'Status'}")
        print("-" * 130)

        evaluable_f1s = []
        evaluable_orcs = []

        for fold in range(1, 7):
            f_str = str(fold)
            m = shrink.get(f_str, {}).get(app, {})
            is_eval = fold in metered_houses[app]

            n_cal = m.get("n_active_calib", 0)
            alpha = m.get("alpha", 0.0)
            th_shrink = m.get("theta_shrink", 0.50)
            f1 = m.get("f1", np.nan)
            prec = m.get("precision", np.nan)
            rec = m.get("recall", np.nan)
            orc = m.get("oracle_f1", np.nan)
            orc_th = m.get("oracle_th", np.nan)
            eval_act = raw_counts[fold][app]["eval_act"]

            if is_eval and not np.isnan(f1):
                status = f"INCLUDED (evaluable)"
                evaluable_f1s.append(f1)
                evaluable_orcs.append(orc)
            elif fold == 5 and app == "dishwasher":
                status = "EXCLUDED (N_eval=0, all 494 act in calib)"
            elif fold == 6 and app == "dishwasher":
                status = "EXCLUDED (degenerate, only 19 act samples)"
            elif fold == 5 and app == "microwave":
                status = "EXCLUDED (virtually unmetered: 1 act sample in 44d)"
            elif fold == 6 and app == "microwave":
                status = "EXCLUDED (unmetered in REDD)"
            elif fold == 4 and app in ["fridge", "microwave"]:
                status = "EXCLUDED (unmetered in REDD)"
            elif fold in [2, 5] and app == "washing_machine":
                status = "EXCLUDED (unmetered in REDD)"
            elif fold == 6 and app == "washing_machine":
                status = "EXCLUDED (negligible signal, 51 act samples)"
            else:
                status = "EXCLUDED"

            f1_str = f"{f1:<10.4f}" if not np.isnan(f1) else f"{'N/A':<10s}"
            prec_str = f"{prec:<10.4f}" if not np.isnan(prec) else f"{'N/A':<10s}"
            rec_str = f"{rec:<10.4f}" if not np.isnan(rec) else f"{'N/A':<10s}"
            orc_str = f"{orc:<10.4f}" if not np.isnan(orc) else f"{'N/A':<10s}"
            orc_th_str = f"({orc_th:<.2f})" if not np.isnan(orc_th) else "(N/A)"

            print(f"Fold {fold:<2} | House {fold:<7} | {n_cal:<12d} | {alpha:<8.4f} | {th_shrink:<12.4f} | {f1_str} {prec_str} {rec_str} | {orc_str} {orc_th_str:<12s} | {eval_act:<10,d} | {status}")

        print("-" * 130)
        mean_f1 = np.mean(evaluable_f1s)
        mean_orc = np.mean(evaluable_orcs)
        v3_base = v3_baselines[app]
        delta = mean_f1 - v3_base
        rel_gain = (delta / v3_base) * 100
        ceil_rec = (mean_f1 / mean_orc) * 100

        print(f"SUMMARY FOR {app.upper()}:")
        print(f"  Evaluable Houses Included    : {metered_houses[app]} ({len(evaluable_f1s)} folds)")
        print(f"  Macro-Averaged Eval F1       : {mean_f1:.4f}")
        print(f"  Oracle Ceiling F1            : {mean_orc:.4f} (Ceiling Recovery: {ceil_rec:.2f}%)")
        print(f"  v3 Baseline F1               : {v3_base:.4f}")
        print(f"  Absolute Improvement (Delta) : {delta:+.4f}")
        print(f"  Relative Gain                : {rel_gain:+.2f}%")

    print("\n" + "=" * 130)
    print("                     INDEPENDENT COMPUTATION SANITY CHECK (SECTION 3)")
    print("=" * 130)
    for app in ["microwave", "dishwasher", "washing_machine"]:
        folds = metered_houses[app]
        f1_list = [shrink[str(f)][app]["f1"] for f in folds if not np.isnan(shrink[str(f)][app]["f1"])]
        mean_val = np.mean(f1_list)
        base_val = v3_baselines[app]
        num = mean_val - base_val
        rel = (num / base_val) * 100
        print(f"{app:<16s}: Mean F1 = {mean_val:.6f}, Base = {base_val:.4f} | Numerator = {num:+.6f}, Denom = {base_val:.4f} -> Relative = {rel:+.4f}%")

if __name__ == "__main__":
    main()
