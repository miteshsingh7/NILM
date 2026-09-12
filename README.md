# Home Energy-Usage Disaggregation (NILM) — Multi-Target Seq2Seq Framework

A deep learning Non-Intrusive Load Monitoring (NILM) framework that disaggregates smart-meter aggregate power into individual appliance consumption profiles using a **single shared encoder with multi-appliance heads**.

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
- **Single Shared Representation**: Learns joint temporal and spectral features across all active household loads.
- **Dual-Branch Heads**: Each appliance head outputs both **continuous power regression** (Watts) and **binary state classification** (probability of ON/OFF state).
- **Active-Period Normalization**: Target loads are normalized using mean and standard deviation computed **exclusively over active periods**, preventing long inactive zero-stretches from crushing the active target distribution.
- **Cross-Household Generalization**: Enforces strict hold-out splits (e.g., REDD House 2) that never participate in training or validation, directly testing domain transfer across homes.

---

## 📂 Repository Structure

```
NILM/
├── README.md                      # Comprehensive documentation
├── requirements.txt               # Dependencies
├── data/
│   ├── raw/                       # Raw REDD / UK-DALE channel dat files
│   └── processed/                 # Resampled (6s) CSV benchmarks
├── src/
│   ├── __init__.py
│   ├── config.py                  # Thresholds, appliances, and hyperparameters
│   ├── data_pipeline.py           # REDD/UK-DALE loading, 6s resampler, windowing, active normalizer
│   ├── sample_data.py             # Realistic multi-appliance synthetic generator
│   ├── model.py                   # PyTorch Conv1D + BiLSTM shared encoder & dual-branch heads
│   ├── loss.py                    # Joint multi-task loss: MSE(power) + λ * BCE(on/off)
│   ├── train.py                   # Mixed-precision (AMP) training loop with checkpointing
│   ├── evaluate.py                # NDE, F1, MAE, SAE, and cross-household comparison
│   ├── event_detection.py         # Standalone on/off state & event scoring module
│   └── utils.py                   # kWh integration, tariff cost calculations, metrics
├── app/
│   ├── dashboard.py               # Streamlit Cost & Usage Attribution Dashboard
│   └── live_demo.py               # Streamlit Real-Time Disaggregation Simulation
├── notebooks/
│   └── 01_nilm_training_kaggle.ipynb # Ready-to-run Kaggle GPU notebook
└── tests/
    ├── test_data_pipeline.py      # Unit tests for preprocessing & windowing
    ├── test_model.py              # Unit tests for forward pass & gradient flow
    ├── test_metrics.py            # Unit tests for NDE, F1, and kWh calculations
    └── test_event_detection.py    # Unit tests for standalone event detection
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
If you haven't downloaded the full multi-gigabyte REDD or UK-DALE datasets yet, generate realistic 6-second smart meter traces:
```bash
python3 -m src.sample_data --houses 3 --days 6 --out_dir data/processed
```
This generates House 1 (train/val), House 2 (unseen held-out generalization test), and House 3 (train/val).

### 3. Train the Model
```bash
python3 -m src.train --epochs 35 --batch_size 128 --held_out_house 2
```
Outputs saved to `checkpoints/`:
- `best_model.pt`: Checkpoint with lowest validation loss.
- `latest_checkpoint.pt`: Checkpoint for training resumption.
- `norm_params.json`: Active-period and mains normalization statistics.
- `history.json`: Training and validation loss curves.

### 4. Evaluate & Cross-Household Generalization
```bash
python3 -m src.evaluate --checkpoint checkpoints/best_model.pt
```
Outputs a markdown comparison table displaying:
- **In-Distribution Test**: Performance on seen houses (unseen time split).
- **Cross-Household Test**: Performance on completely unseen held-out House 2.
- **Generalization Gap**: $\Delta \text{NDE}$ and $\Delta \text{F1}$.

### 5. Launch Cost & Usage Dashboard (Streamlit)
```bash
streamlit run app/dashboard.py
```
- Interactive ₹/kWh electricity tariff calculator.
- Stacked area power breakdown.
- Financial cost attribution per appliance.
- Duty cycle and active hours summary table.

### 6. Launch Real-Time Disaggregation Demo (Streamlit)
```bash
streamlit run app/live_demo.py
```
- Simulates a live IoT smart meter feed.
- Rolling $L=599$ sample buffer.
- Live glowing ON/OFF status LEDs and instant power gauges.

---

## 🧪 Running Unit Tests
```bash
python3 -m pytest tests -v
```
All unit tests verify:
- 6-second resampling and gap forward-filling.
- Active-period normalization without zero-bias distortion.
- Fixed-length sliding window tensor shapes ($L=599$, stride 149 vs 599).
- Gradient flow through the shared encoder and all independent heads.
- NDE, F1, SAE, and energy integration calculations.

---

## 📊 Evaluation Metrics

1. **Normalized Disaggregation Error (NDE)**:
   $$\text{NDE} = \sqrt{\frac{\sum_t (y_t - \hat{y}_t)^2}{\sum_t y_t^2}}$$
   Computed on physical de-normalized Watts.
2. **On/Off F1 Score**:
   $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
   Evaluated at threshold $0.5$ against binary ground-truth states.
3. **Energy Consumption Integration**:
   $$\text{kWh} = \sum_{t} P_t \times \frac{\Delta t}{3600 \times 1000}$$
   where $\Delta t = 6\,\text{seconds}$.
4. **Monetary Cost**:
   $$\text{Cost} = \text{kWh} \times \text{Tariff} \quad (\text{e.g. ₹7.50/kWh})$$
