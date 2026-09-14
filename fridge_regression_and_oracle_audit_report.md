# Follow-Up Forensic Investigation: Fridge Regression & Oracle F1 Disambiguation

**Investigation Date:** September 14, 2026  
**Artifact Path:** `fridge_regression_and_oracle_audit_report.md`  
**Referenced Checkpoints:**
- v3 Baseline: `checkpoints/loho_cv_decoupled/`
- Combined Run: `checkpoints/loho_cv_combined_phase7/`
**Target Scope:** Houses 5 & 6 Refrigerator collapse, Oracle F1 duality, windowing geometry, and House 5 Dishwasher cycle ground truth.

---

## 1. Recovery of Original v3-Era House 5 & House 6 Fridge Numbers

The v3-era baseline evaluation logs were located in `checkpoints/loho_cv_decoupled/few_shot_calibration_results.json` and the per-fold `eval_results.json` files. 

### Source Log Verification (`few_shot_calibration_results.json`)
The exact per-fold 24-hour calibration baseline that produced the **0.4697** mean across 5 active folds:
```json
{
  "window_hours": 24,
  "mean_f1": 0.4697,
  "mean_recall": 0.6627,
  "mean_precision": 0.3843,
  "oracle_ceiling_f1": 0.4827,
  "always_on_f1": 0.3502,
  "folds": {
    "1": {
      "tau": 0.35,
      "f1": 0.4365,
      "recall": 0.7261,
      "precision": 0.312,
      "oracle_f1": 0.4691,
      "always_on_f1": 0.3438
    },
    "2": {
      "tau": 0.25,
      "f1": 0.6159,
      "recall": 0.6145,
      "precision": 0.6174,
      "oracle_f1": 0.6142,
      "always_on_f1": 0.4485
    },
    "3": {
      "tau": 0.4,
      "f1": 0.071,
      "recall": 0.245,
      "precision": 0.0415,
      "oracle_f1": 0.1027,
      "always_on_f1": 0.0604
    },
    "5": {
      "tau": 0.07,
      "f1": 0.5978,
      "recall": 0.8833,
      "precision": 0.4518,
      "oracle_f1": 0.599,
      "always_on_f1": 0.416
    },
    "6": {
      "tau": 0.225,
      "f1": 0.6271,
      "recall": 0.8445,
      "precision": 0.4987,
      "oracle_f1": 0.6284,
      "always_on_f1": 0.4824
    }
  }
}
```

### Side-by-Side Comparison: v3 Baseline vs Combined Architecture

| House | v3 24h Calibrated F1 | v3 Optimal $\tau$ | v3 Uncalibrated F1 (@0.50) | Combined Run Shrinkage F1 | Combined Run Uncalibrated F1 (@0.50) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **House 1** | **0.4365** | 0.350 | 0.1446 | **0.4703** | 0.4129 | **Improved (+0.0338)** |
| **House 2** | **0.6159** | 0.250 | 0.1235 | **0.6891** | 0.6426 | **Improved (+0.0732)** |
| **House 3** | **0.0710** | 0.400 | 0.0643 | **0.0717** | 0.0743 | **Parity (+0.0007)** |
| **House 5** | **0.5978** | 0.070 | 0.0286 | **0.0000** | 0.0000 | **Real Regression (-0.5978)** |
| **House 6** | **0.6271** | 0.225 | 0.0019 | **0.0000** | 0.0000 | **Real Regression (-0.6271)** |
| **5-Fold Mean** | **0.4697** | — | **0.0727** | **0.2462** | **0.2260** | **-0.2235 (Regression)** |

### Answer to Question 1
**Houses 5 and 6 were NOT always weak on fridge.** In fact, in the v3 baseline, House 5 (F1 = **0.5978**) and House 6 (F1 = **0.6271**) were the two **highest-scoring folds** in the entire benchmark. The 0.4697 mean was *not* propped up by Houses 1–3; rather, Houses 5 and 6 carried the mean. The zeroing-out of Houses 5 and 6 in the combined run is a **real regression**, not an artifact of unweighted averaging.

---

## 2. Root Cause: Why Three Thresholds Zeroed Out on Houses 5 & 6

The investigation audited the weight matrices, raw prediction distributions, and evaluation routines across devices (Apple Silicon MPS vs CPU) for `checkpoints/loho_cv_combined_phase7/fold_5/best_model.pt` and `fold_6/best_model.pt`.

### 2A. Discovery of Corrupted (NaN) Model Weights in Folds 5 & 6
Direct tensor inspection revealed that during the 35-epoch training of Fold 5 and Fold 6, numerical instability caused gradient explosion:
- **Fold 1 Checkpoint**: 0 / 76 weight tensors contain NaN.
- **Fold 2 Checkpoint**: 0 / 76 weight tensors contain NaN.
- **Fold 3 Checkpoint**: 0 / 76 weight tensors contain NaN.
- **Fold 4 Checkpoint**: 0 / 76 weight tensors contain NaN.
- **Fold 5 Checkpoint**: **53 / 76 weight tensors contain NaN** (`conv1.weight`, `bn1.weight`, `lstms.fridge.*`, `heads.fridge.*`).
- **Fold 6 Checkpoint**: **54 / 76 weight tensors contain NaN**.

### 2B. Device Backend Discrepancy: MPS Kernel Behavior vs CPU IEEE 754
The model's behavior was obscured because the pipeline alternated between two different execution devices:

1. **On Apple Silicon MPS (`mps:0`)**:
   - PyTorch's Metal Performance Shaders (MPS) linear algebra kernels handle NaN weight matrices without throwing an error and without propagating IEEE 754 `NaN`.
   - Instead, the corrupted layers collapse to a **static, constant scalar output** for all timesteps:
     - **House 5 Fridge Output**: $\hat{p} = \mathbf{0.239148}$ identically for all 599 timesteps of every window.
     - **House 6 Fridge Output**: $\hat{p} = \mathbf{0.243467}$ identically for all 599 timesteps of every window.

2. **On CPU (`map_location="cpu"`)**:
   - The same checkpoint loaded on CPU produces standard IEEE 754 `NaN` for all output activations.

### 2C. Explaining the Three Conflicting Outcomes
This backend duality and the static probability output explain the observed behavior:

1. **Why static $\tau = 0.50$ produced Recall = 0.0000, F1 = 0.0000**:
   - On MPS, the model outputs $\hat{p} = 0.239148$. Since $0.239148 < 0.50$, every timestep is predicted as **OFF**.
   - $\text{TP} = 0$, $\text{FN} = 8,123 \implies \text{Recall} = 0.0000, \text{F1} = 0.0000$.

2. **Why $P_{10}$ adapted threshold ($\tau = 0.6496$) produced Recall = 0.0000, F1 = 0.0000**:
   - The unsupervised shift increased the threshold from $0.50$ to $0.6496$ due to House 5's high mains baseload ($180.6$ W).
   - Since $0.239148 < 0.6496$, every timestep is predicted as **OFF**.

3. **Why the Full-House Oracle Sweep Claimed F1 = 0.3989 at $\theta^* \approx 0.01$**:
   - In `run_evaluation()` on MPS, the oracle sweeps thresholds from 0.01 to 0.99.
   - For any threshold $\theta \le 0.2391$ (such as 0.01 or 0.10), $\hat{p} = 0.239148 \ge \theta$ predicts **ALL ON** for the entire dataset.
   - House 5 test set contains 11,043 active samples out of 44,326 total timesteps ($\text{Prevalence} = 0.2491$).
   - When predicting All-ON:
     $$\text{Precision} = 0.2491, \quad \text{Recall} = 1.0000, \quad \text{F1} = \frac{2 \times 0.2491 \times 1.0}{1.0 + 0.2491} = \mathbf{0.3989}$$
   - **The claimed "Oracle F1 = 0.3989" was simply the trivial All-ON majority baseline.**

4. **Why James-Stein Shrinkage Calibration Produced F1 = 0.0000 at $\theta = 0.1018$**:
   - In `run_combined_phase7_end_to_end.py` line 227:
     ```python
     raw_model = DecoupledTemporalNILM(...)
     state = torch.load(best_ckpt_path, map_location="cpu")["model_state_dict"]
     raw_model.load_state_dict(clean_st)
     raw_model.eval()
     ```
   - The model was loaded onto **CPU** without transferring it back to MPS (`.to(device)` was omitted).
   - On CPU, the NaN weights output strict IEEE 754 `NaN`s.
   - In NumPy, `NaN >= 0.1018` evaluates to `False` for every single element.
   - Hence, $\text{TP} = 0$, $\text{FP} = 0$, $\text{FN} = 8,123$, $\text{TN} = 23,624 \implies \mathbf{\text{F1} = 0.0000}$.

### 2D. Confusion Matrix Verification (Post-24h Eval Windows)

#### House 5 Fridge (Evaluated with MPS Predictions: $\hat{p} = 0.239148$ everywhere)
| Threshold $\theta$ | TP | FP | FN | TN | Precision | Recall | F1 Score | Notes |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $\mathbf{0.5000}$ | 0 | 0 | 8,123 | 23,624 | 0.0000 | 0.0000 | **0.0000** | All Predicted Inactive |
| $\mathbf{0.2000}$ | 8,123 | 23,624 | 0 | 0 | 0.2559 | 1.0000 | **0.4075** | All Predicted Active (Trivial All-ON) |
| $\mathbf{0.1018}$ | 8,123 | 23,624 | 0 | 0 | 0.2559 | 1.0000 | **0.4075** | All Predicted Active (Trivial All-ON) |
| $\mathbf{0.0500}$ | 8,123 | 23,624 | 0 | 0 | 0.2559 | 1.0000 | **0.4075** | All Predicted Active (Trivial All-ON) |

*(Note: When evaluated on CPU where predictions are NaNs, TP=0, FP=0 across all thresholds, yielding F1=0.0000).*

#### House 6 Fridge (Evaluated with MPS Predictions: $\hat{p} = 0.243467$ everywhere)
| Threshold $\theta$ | TP | FP | FN | TN | Precision | Recall | F1 Score | Notes |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $\mathbf{0.5000}$ | 0 | 0 | 82,836 | 178,927 | 0.0000 | 0.0000 | **0.0000** | All Predicted Inactive |
| $\mathbf{0.2000}$ | 82,836 | 178,927 | 0 | 0 | 0.3165 | 1.0000 | **0.4808** | All Predicted Active (Trivial All-ON) |
| $\mathbf{0.1000}$ | 82,836 | 178,927 | 0 | 0 | 0.3165 | 1.0000 | **0.4808** | All Predicted Active (Trivial All-ON) |

---

## 3. Disambiguation of Oracle F1 Definitions & Window Generation Audit

The previous report conflated two distinct quantities under the label "Oracle F1":
1. `oracle_f1_full_house`: The optimal F1 achievable by searching $\theta \in [0.01, 0.99]$ across the **entire held-out test series** (Day 0 to End).
2. `oracle_f1_post24h_suffix`: The optimal F1 achievable on the **strictly held-out post-24h evaluation suffix** ($[24\text{h}:\text{end}]$), reserving Day 1 exclusively for commissioning calibration.

### Window Generation Parameters Across Protocols

| House | Total Record Duration | Protocol Full-House (`test_ds`) | Protocol B Calibration (`calib_x`) | Protocol B Evaluation (`eval_x`) | Window Count Discrepancy Rationale |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **House 1** | 522,350 samples (36.3 d) | Stride=599, Overlap=0%<br>**771 windows** (95,172 act) | Stride=149, Overlap=75%<br>**89 windows** (8,638 act) | Stride=599, Overlap=0%<br>**750 windows** (93,231 act) | $89 + 750 = 839 \ne 771$. Calibration used 75% overlap on Day 1 to maximize calibration events. |
| **House 2** | 504,409 samples (35.0 d) | Stride=599, Overlap=0%<br>**323 windows** (55,351 act) | Stride=149, Overlap=75%<br>**68 windows** (10,335 act) | Stride=599, Overlap=0%<br>**307 windows** (53,047 act) | $68 + 307 = 375 \ne 323$. Calibration used 75% overlap on Day 1. |
| **House 3** | 645,086 samples (44.8 d) | Stride=599, Overlap=0%<br>**392 windows** (7,098 act) | Stride=149, Overlap=75%<br>**46 windows** (425 act) | Stride=599, Overlap=0%<br>**380 windows** (6,994 act) | $46 + 380 = 426 \ne 392$. Calibration used 75% overlap on Day 1. |
| **House 4** | 690,895 samples (48.0 d) | Stride=599, Overlap=0%<br>**592 windows** (0 act) | Stride=149, Overlap=75%<br>**73 windows** (0 act) | Stride=599, Overlap=0%<br>**572 windows** (0 act) | $73 + 572 = 645 \ne 592$. Unmetered fridge/microwave. |
| **House 5** | 631,164 samples (43.8 d) | Stride=599, Overlap=0%<br>**74 windows** (11,043 act) | Stride=149, Overlap=75%<br>**80 windows** (11,188 act) | Stride=599, Overlap=0%<br>**53 windows** (8,123 act) | $80 + 53 = 133 \ne 74$. $14,400 \pmod{599} = 24$ sample boundary offset + 75% overlap in calib. |
| **House 6** | 337,125 samples (23.4 d) | Stride=599, Overlap=0%<br>**460 windows** (85,556 act) | Stride=149, Overlap=75%<br>**88 windows** (9,793 act) | Stride=599, Overlap=0%<br>**437 windows** (82,836 act) | $88 + 437 = 525 \ne 460$. Calibration used 75% overlap on Day 1. |

### Confirmation of Prediction / Window Set Sharing
- **Full House Test Evaluation** runs over `test_ds` (non-overlapping stride 599 starting at $t=0$).
- **Commissioning Shrinkage Evaluation** runs over `eval_x` (non-overlapping stride 599 starting at $t=14,400$).
- Because $14,400 \pmod{599} = 24$, the window boundaries of `eval_x` are shifted by 24 timesteps relative to `test_ds[24:]`. This slight phase shift alters the slicing of edge transitions, explaining the minor difference between `oracle_f1_full_house` (0.3989 on H5) and `oracle_f1_post24h_suffix` (0.4075 on H5).
- In all future documentation, `oracle_f1_full_house` and `oracle_f1_post24h_suffix` are separated explicitly.

---

## 4. Correction of the House 5 Dishwasher Exclusion Rationale

Section 3 of the previous report incorrectly attributed House 5's dishwasher exclusion to "corrupted logger flatline." Ground-truth verification demonstrates this was inaccurate:

### Ground-Truth Timestamp Analysis
Scanning raw `data/processed/redd_real_house_5.csv` for `dishwasher >= 10.0 W`:
- **Total Valid Timesteps**: 52,641
- **Total Active Timesteps ($\ge 10$ W)**: **494 samples**
- **Day 1 Window $[0:24\text{h}]$ (`2011-04-18 04:24:00` to `2011-04-19 04:24:00 UTC`)**:
  - Contains **all 494 active samples**.
  - Cycle start: `2011-04-18 23:43:18 UTC`
  - Cycle end: `2011-04-19 00:40:30 UTC` (duration: 57 minutes, peak power: 529.5 W).
  - Clean, uncorrupted, classic two-stage dishwasher profile with high SNR.
- **Post-24h Evaluation Window $[24\text{h}:\text{end}]$ (`2011-04-19 04:24:00` to `2011-06-01 00:20:18 UTC`)**:
  - Total valid samples: 38,520
  - **Active samples: EXACTLY 0**.

### Corrected Record
House 5 dishwasher was **not excluded due to noisy, degenerate, or flatlined sensor data**. The signal was high-quality and strongly separated. The exclusion is strictly methodological: **the house's single dishwasher cycle fell entirely inside the 24-hour commissioning window**, leaving zero ground-truth positive events in the post-24h held-out evaluation suffix ($[24\text{h}:\text{end}]$). Computing precision, recall, or F1 on the held-out suffix is therefore mathematically undefined ($\frac{0}{0}$).

---

## 5. Summary of Conclusions & Action Items

1. **The Fridge Regression is Real**: Houses 5 and 6 did not fail because they were "historically weak" — they were the highest-scoring folds in v3 ($0.5978$ and $0.6271$).
2. **Failure Mechanism Identified**: The combined architecture encountered gradient explosion in Folds 5 and 6 during MPS training, corrupting 53/76 and 54/76 weight tensors into NaNs. On MPS, this produced a static constant probability output ($0.2391$ and $0.2435$), which collapsed under all standard thresholds. On CPU, it evaluated to IEEE 754 NaNs, producing zero true positives.
3. **The Discrepancies are Resolved**:
   - `oracle_f1_full_house` and `oracle_f1_post24h_suffix` have been disambiguated. The "0.3989" was the trivial All-ON baseline of the static MPS output.
   - Window count differences ($839$ vs $771$) are formally documented as the consequence of 75% overlapping calibration stride ($S=149$) on Day 1 vs non-overlapping stride ($S=599$) on full test.
   - House 5 Dishwasher is correctly classified as "untestable on $[24\text{h}:\text{end}]$ due to cycle placement entirely within Day 1," not "corrupted data."
