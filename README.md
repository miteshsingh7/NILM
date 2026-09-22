# Home Energy-Usage Disaggregation (NILM) — Multi-Target Seq2Seq Framework

A deep learning Non-Intrusive Load Monitoring (NILM) framework that disaggregates smart-meter aggregate power into individual appliance consumption profiles using a **single shared encoder with multi-appliance heads**.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-46%20passed-brightgreen.svg)](tests/)
[![Live Deployment](https://img.shields.io/badge/Render-Live%20Telemetry%20Rack-success.svg)](https://nilm-telemetry-rack.onrender.com)

---

## ⚡ Architectural Overview

Unlike traditional NILM workflows that train $N$ separate models (one per appliance), this framework uses a **multi-target sequence-to-sequence** architecture trained in a single pass:

```
                          ┌───────────────────────────┐
                          │ Aggregate Mains (L=599, 1)│
                          └─────────────┬─────────────┘
                                        │
                         ┌──────────────▼──────────────┐
                         │      SHARED ENCODER         │
                         │ Conv1D(32, k=9) + BatchNorm │
                         │ Conv1D(64, k=7) + BatchNorm │
                         │ Conv1D(128, k=5) + BatchNorm│
                         │ Dropout(0.2)                │
                         │ Bidirectional LSTM (128)    │
                         └──────────────┬──────────────┘
                                        │ (B, 599, 256)
        ┌───────────────────┬───────────┴───────────┬───────────────────┐
        │                   │                       │                   │
 ┌──────▼──────┐     ┌──────▼──────┐         ┌──────▼──────┐     ┌──────▼──────┐
 │ Fridge Head │     │Microwave Head│        │Dishwasher Hd│     │WashingMach. │
 └──────┬──────┘     └──────┬──────┘         └──────┬──────┘     └──────┬──────┘
        ├─ Power (Watts)    ├─ Power (Watts)        ├─ Power (W)        ├─ Power (W)
        └─ On/Off State     └─ On/Off State         └─ On/Off State     └─ On/Off State
```

### Key Technical Innovations
- **Single Shared Representation**: Learns joint temporal and spectral representations across all active household loads in one 256-dimensional feature vector, cutting training compute by ~75% compared to 4 separate models.
- **Dual-Branch Heads**: Each appliance head outputs both **continuous power regression** (Watts) and **binary state classification** (probability of ON/OFF state).
- **Active-Period Normalization**: Target loads are normalized using mean and standard deviation computed **exclusively over active periods**, preventing long inactive zero-stretches from crushing the active target distribution.
- **Decoupled Weighted Loss**: Regression loss applies per-appliance active-state upweighting (fridge: 1.44x, microwave/dishwasher/washing machine: 8.0x) inversely proportional to class priors. Loss is ungated during training to avoid degenerate minima, applying the probability gate only at evaluation time.
- **Active-Window Oversampling**: Boosts under-represented intermittent appliances (`boost_weight = 2.5`) during training batches via `WeightedRandomSampler` while keeping validation and test sets strictly at natural distributions.
- **Strict 6-Fold Leave-One-House-Out Cross-Validation (LOHO-CV)**: Benchmarked across all 6 REDD houses, evaluating true cross-household domain generalization rather than a single fixed train/test split.
- **24-Hour Chronological Few-Shot Threshold Calibration**: Fixes cross-household sigmoid confidence suppression by recalibrating decision thresholds $\tau^*$ on the first 24 hours of an unseen home without fine-tuning frozen weights—recovering **>97% of the fridge oracle F1 ceiling** and raising dishwasher F1 from **0.1601 to 0.3625**.

---

## 📊 Benchmark Results & Cross-Household Generalization

Every metric reported below is evaluated under strict **6-Fold Leave-One-House-Out Cross-Validation (LOHO-CV)** on the REDD dataset (where each test home is completely unseen during training).

### 1. Primary 6-Fold LOHO-CV Benchmark (Adopted Baseline)

Evaluated across all 6 folds. **Macro-F1** is the mean of each fold's independent F1 score ($1/K \sum F_{1,k}$), reported alongside **Harmonic-F1** ($2 \bar{P} \bar{R} / (\bar{P} + \bar{R})$) and the theoretical **Oracle F1 Ceiling**:

| Appliance | Precision | Recall | Macro F1 | Harmonic F1 | Oracle F1 Ceiling | Ceiling Recovery | Calibration Protocol | Audit Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Refrigerator** | 0.3843 | 0.6627 | **0.4697** | 0.4865 | 0.4827 | **97.31%** | 24h Few-Shot ($\tau^*=0.20$) | `CALIBRATED_OPTIMAL` |
| **Microwave** | 0.4790 | 0.4120 | **0.3986** | 0.4430 | 0.4120 | 96.75% | Fixed ($\tau=0.50$) | `VALIDATED_CROSS_HOUSEHOLD` |
| **Dishwasher** | 0.3577 | 0.5144 | **0.3625** | 0.4220 | 0.3981 | **91.06%** | 24h Few-Shot ($\tau^* \approx 0.23$) | `CALIBRATED_OPTIMAL` |
| **Washing Machine** | 0.3980 | 0.1762 | **0.2334** | 0.2443 | 0.2450 | 95.27% | Fixed ($\tau=0.50$) | `VALIDATED_CROSS_HOUSEHOLD` |

> [!NOTE]
> - **Jensen's Inequality Gap**: Harmonic-F1 sits above Macro-F1 because Macro-F1 computes the average of ratios, whereas Harmonic-F1 computes the ratio of averages across heterogeneous residential homes.
> - **Dishwasher Evaluable Folds**: Evaluated across Houses 1–4. House 6 has only 12s of recorded activity across 19 days, and House 5's single active cycle falls inside the 24h calibration window, leaving 0 positive evaluation samples in its $[24\text{h}:\text{end}]$ span.

---

### 2. Transfer Learning Experiment (UK-DALE Pretrain → REDD Fine-Tune)

A full 6-fold LOHO-CV transfer learning experiment was tested with identical 24h calibration. The approach **did not survive validation and was rejected**:

| Appliance | No-Transfer Baseline | With UK-DALE Transfer | Relative Change | Outcome & Decision |
| :--- | :---: | :---: | :---: | :--- |
| **Refrigerator** | 0.4697 | 0.4700 | +0.06% | Statistically identical (noise floor) |
| **Microwave** | 0.3986 | 0.3202 | −19.66% | **Severe negative transfer** |
| **Dishwasher** | 0.3625 | 0.3294 | −9.15% | **Severe negative transfer** |
| **Washing Machine** | 0.2334 | 0.2038 | −12.66% | **Negative transfer** |

*Verdict*: While an isolated House 2 test appeared promising, full 6-fold evaluation revealed severe negative transfer across multiple unseen households. Transfer learning was dropped from the adopted production pipeline.

---

### 3. Phase 7 Architecture Comparison (GroupNorm + DecoupledTemporal + Protocol B)

An advanced architecture variant featuring GroupNorm, `DecoupledTemporalNILM`, and James-Stein shrinkage calibration (Protocol B) was benchmarked against the adopted v3 baseline on identical evaluable houses:

| Appliance | Same-House v3 Baseline F1 | Phase 7 F1 | Relative Gain | Oracle Ceiling | Audit Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Refrigerator** | 0.4697 | **0.5005** | **+6.56%** | 0.5221 (95.86% recovery) | `VERIFIED_GAIN` |
| **Dishwasher** | 0.3625 | **0.4787** | **+32.05%** | 0.4920 (97.30% recovery) | `VERIFIED_GAIN` |
| **Microwave** | 0.5315* | 0.5250 | −1.22% | 0.5983 (87.75% recovery) | `NOT_VALIDATED` |
| **Washing Machine** | 0.3112* | 0.3096 | −0.51% | 0.3443 (89.92% recovery) | `NOT_VALIDATED` |

*\*Same-house baselines recomputed on the exact evaluable houses used by Phase 7 (Houses 1, 2, 3 for Microwave; Houses 1, 3, 4 for Washing Machine) rather than the unrestricted 6-fold mean with zero-signal houses.*

---

## 🖥️ Live Telemetry Dashboard & Production Deployment

The project features a **custom asynchronous aiohttp WebSocket telemetry server** paired with a Google Stitch-designed **"DSP Telemetry Rack"** oscilloscope-style interface:

- **10 Hz Telemetry Stream**: Streams real-time aggregate-to-per-appliance disaggregation over WebSockets (`ws://.../ws`).
- **Composite Oscilloscope**: Multi-channel waveform display featuring raw aggregate mains and 4 stacked appliance traces with auto-scaling divisions.
- **Inference Latency**: Sub-50ms CPU execution (**34.6 ms** standard PyTorch CPU, **4.8 ms** quantized).
- **Interactive REST Endpoints**:
  - `/api/status`: Health check, uptime, memory, active WebSockets.
  - `/api/feeders`: Feeder switching across Houses 1, 2, 3, 4, and 6.
  - `/api/usage_cost`: Running monetary cost attribution supporting $, ₹, £, and € (defaults to ₹7.50/kWh for INR, $0.15/kWh for USD/EUR).
  - `/api/signatures`: Power and duty-cycle reference signatures.
  - `/api/sensor_feeds`: Emulated ADC metering stack (RMS voltage, power factor, THD).
  - `/api/diagnostics`: Live model governance view exposing Macro-F1, Harmonic-F1, Oracle Ceilings, and validation status tags.
  - `/api/export`: CSV telemetry export for offline auditing.

### 🌐 Public Deployment
The application is deployed publicly as a containerized Docker web service on Render:
- **Live URL**: [https://nilm-telemetry-rack.onrender.com](https://nilm-telemetry-rack.onrender.com)
- **Live Diagnostics API**: [https://nilm-telemetry-rack.onrender.com/api/diagnostics](https://nilm-telemetry-rack.onrender.com/api/diagnostics)

*(Note: Render free tier services spin down after 15 minutes of inactivity; initial cold start takes ~34 seconds).*

---

## 📂 Repository Structure

```
NILM/
├── README.md                      # Comprehensive documentation and benchmark report
├── Dockerfile                     # Production container image (PyTorch CPU + aiohttp)
├── render.yaml                    # Render Blueprint deployment configuration
├── requirements.txt               # Base dependencies
├── requirements-docker.txt        # Slim container dependencies
├── run_dashboard.py               # Production launcher for aiohttp Telemetry Rack server
├── app/
│   ├── server.py                  # Asynchronous aiohttp WebSocket server & REST API
│   ├── dashboard.py               # Legacy Streamlit Cost & Usage Dashboard
│   └── live_demo.py               # Legacy Streamlit Real-Time Disaggregation Demo
├── stitch_dashboard/
│   ├── index.html                 # DSP Telemetry Rack oscilloscope UI (Google Stitch design)
│   └── screen.png                 # Oscilloscope UI screenshot
├── data/
│   ├── raw/                       # Raw REDD / UK-DALE dat files
│   └── processed/                 # Resampled (6s) CSV benchmarks and feeder snippets
├── checkpoints/
│   ├── fold_1/ ... fold_6/        # Checkpoints for each LOHO-CV fold
│   ├── loho_cv_decoupled/         # Decoupled benchmark results & calibration JSONs
│   │   ├── few_shot_calibration_results.json
│   │   └── dishwasher_calibration_results.json
│   └── loho_cv_combined_phase7/   # Phase 7 retrained architecture summary JSONs
├── src/
│   ├── __init__.py
│   ├── config.py                  # Thresholds, channel maps, and on_weight loss config
│   ├── data_pipeline.py           # REDD/UK-DALE loading, 6s resampler, windowing, normalizer
│   ├── sample_data.py             # Realistic multi-appliance synthetic generator
│   ├── model.py                   # Shared Conv1D + BiLSTM encoder & dual-branch heads
│   ├── loss.py                    # Decoupled MSE + BCE loss with active-state upweighting
│   ├── train.py                   # Mixed-precision (AMP) training loop with gradient clipping
│   ├── loho_cv.py                 # Automated 6-fold Leave-One-House-Out CV runner
│   ├── evaluate.py                # NDE, F1, MAE, SAE, and cross-household comparison
│   ├── event_detection.py         # Standalone on/off state & event scoring module
│   └── utils.py                   # kWh integration, tariff cost calculations, checkpoint guards
├── tests/                         # Full 8-file test suite (46 passed cases)
│   ├── test_data_pipeline.py      # Resampling, windowing, and channel mapping tests
│   ├── test_model.py              # Forward pass, gradient flow, and parameter count
│   ├── test_metrics.py            # NDE, F1, and kWh energy integration
│   ├── test_event_detection.py    # Event detection and hysteresis filtering
│   ├── test_server.py             # aiohttp REST routes and WebSocket lifecycle
│   ├── test_diagnostics.py        # /api/diagnostics payload and metric validation
│   ├── test_training_guards.py    # NaN parameter guards and gradient clipping
│   └── test_phase1_pipeline.py    # Pipeline normalization and loss masking tests
└── DISHWASHER_CALIBRATION_AND_MULTIAPPLIANCE_PROTOCOL_B_REPORT.md # Detailed audit reports
```

---

## 🚀 Quick Start

### 1. Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate Benchmark Dataset (Offline Testing)
If working without raw multi-gigabyte REDD archives, generate realistic 6-second smart meter traces:
```bash
python3 -m src.sample_data --houses 3 --days 6 --out_dir data/processed
```

### 3. Launch Production Telemetry Rack (aiohttp + WebSockets)
```bash
python run_dashboard.py --host 0.0.0.0 --port 10000
```
Open **`http://localhost:10000`** to view the live 10 Hz oscilloscope telemetry rack, switch feeders, adjust tariffs, and audit model diagnostics.

### 4. Run Model Training
Train a single held-out fold (e.g., holding out House 2 for cross-household testing):
```bash
python3 -m src.train --epochs 35 --batch_size 128 --held_out_house 2
```
Outputs saved to `checkpoints/fold_2/`:
- `best_model.pt`: Checkpoint with lowest validation loss.
- `latest_checkpoint.pt`: Checkpoint for training resumption.
- `norm_params.json`: Active-period and mains normalization statistics.
- `history.json`: Training and validation loss curves.

### 5. Automated 6-Fold Leave-One-House-Out CV (LOHO-CV)
To run or evaluate all 6 folds across REDD:
```bash
python3 -m src.loho_cv --all_folds --epochs 35
```

### 6. Legacy Streamlit Dashboards (Optional)
```bash
streamlit run app/dashboard.py    # Usage & Cost Attribution
streamlit run app/live_demo.py     # Disaggregation Simulation
```

---

## 🧪 Running Unit Tests

Execute the 8-file automated test suite covering pipeline, model forward pass, server routes, and training guards:
```bash
pytest -q
```
```text
..............................................                           [100%]
46 passed in 3.92s
```

All 46 unit tests verify:
- 6-second resampling and gap forward-filling.
- Active-period normalization without zero-bias distortion.
- Sliding window shapes ($L=599$, stride 149) and channel mapping integrity.
- Gradient clipping and NaN parameter detection guards.
- NDE, F1, SAE, harmonic F1, and energy integration calculations.
- Asynchronous aiohttp WebSocket broadcast and REST endpoint responses.

---

## 📊 Evaluation Metrics

1. **Normalized Disaggregation Error (NDE)**:
   $$\text{NDE} = \sqrt{\frac{\sum_t (y_t - \hat{y}_t)^2}{\sum_t y_t^2}}$$
   Computed on physical de-normalized Watts.
2. **On/Off F1 Score**:
   $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
   Evaluated at decision threshold $\tau = 0.50$ (or calibrated threshold $\tau^*$).
3. **Harmonic Mean F1**:
   $$\bar{F}_1 = 2 \cdot \frac{\bar{P} \cdot \bar{R}}{\bar{P} + \bar{R}}$$
4. **Energy Consumption Integration**:
   $$\text{kWh} = \sum_{t} P_t \times \frac{\Delta t}{3600 \times 1000}$$
   where $\Delta t = 6\,\text{seconds}$.
5. **Monetary Cost**:
   $$\text{Cost} = \text{kWh} \times \text{Tariff} \quad (\text{e.g. ₹7.50/kWh or \$0.15/kWh})$$
