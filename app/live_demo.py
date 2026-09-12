"""Streamlit Real-Time NILM Disaggregation Streaming Simulation.

Simulates an IoT smart-meter gateway replaying real REDD aggregate power readings at
accelerated speed (e.g., 1 hour per 5 seconds), passing sliding windows through the
trained Conv1D-BiLSTM model, and updating live per-appliance power charts in real time.

Features:
- Real REDD dataset replay (House 2 held-out generalization, House 3 active washer, House 1).
- Configurable accelerated speed controls (120x to 3600x, default 720x = 1 hour / 5 seconds).
- Simple Play / Pause / Reset / Step controls.
- Rolling window inference (L=599 samples ~ 1 hour) with the trained PyTorch model.
- Live rolling power chart, instant ON/OFF status badges, and ground truth overlay.
"""

from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

# Ensure project root is in sys.path regardless of launch method or Streamlit runner
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.config import DEFAULT_APPLIANCES, DEFAULT_APPLIANCE_THRESHOLDS
from src.data_pipeline import NormalizationParams
from src.event_detection import evaluate_appliance_events
from src.model import MultiApplianceNILM


# Set page layout
st.set_page_config(
    page_title="NILM Real-Time Streaming Disaggregation Demo",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for glowing badges and sleek meters
st.markdown(
    """
    <style>
    .metric-box {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #dee2e6;
        text-align: center;
    }
    .badge-on {
        color: #28a745;
        font-weight: bold;
        font-size: 1.1em;
    }
    .badge-off {
        color: #6c757d;
        font-weight: normal;
        font-size: 1.1em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_trained_model(checkpoint_path: str = "checkpoints/best_model.pt"):
    """Loads model weights and normalization parameters from checkpoint."""
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.is_absolute():
        ckpt_file = ROOT_DIR / ckpt_file
    if not ckpt_file.exists():
        return None, None, DEFAULT_APPLIANCES

    ckpt = torch.load(str(ckpt_file), map_location="cpu")
    appliances = ckpt.get("appliances", DEFAULT_APPLIANCES)
    norm_dict = ckpt.get("norm_params")
    if norm_dict:
        norm_params = NormalizationParams(
            mains_mean=norm_dict["mains_mean"],
            mains_std=norm_dict["mains_std"],
            appliance_stats=norm_dict["appliance_stats"],
        )
    else:
        norm_file = ckpt_file.parent / "norm_params.json"
        if norm_file.exists():
            norm_params = NormalizationParams.load_json(norm_file)
        else:
            return None, None, appliances

    model = MultiApplianceNILM(appliances=appliances)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, norm_params, appliances


@st.cache_data
def load_house_stream_data(house_id: int) -> pd.DataFrame:
    """Loads real REDD smart-meter time series."""
    file_path = ROOT_DIR / f"data/processed/redd_real_house_{house_id}.csv"
    if file_path.exists():
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        return df
    return pd.DataFrame()


def main():
    st.title("🔴 Live Smart-Meter Disaggregation Simulation")
    st.markdown(
        "**Real-Time Edge Gateway Simulator** | Replays real REDD smart-meter aggregate power at accelerated speed, "
        "maintaining a rolling buffer of $L=599$ samples (~1 hour) through the shared multi-target neural network."
    )
    st.markdown("---")

    # Load model and checkpoint
    model, norm_params, appliances = load_trained_model("checkpoints/best_model.pt")
    if model is None:
        st.error("⚠️ Trained checkpoint not found at `checkpoints/best_model.pt`. Train model first with `src.train`.")
        return

    # --- Sidebar Controls ---
    st.sidebar.header("🎛️ Simulation Controls")

    # 1. House Selection
    house_choice = st.sidebar.selectbox(
        "Data Feed Source",
        options=[
            "REDD House 2 (Held-Out Unseen Home)",
            "REDD House 3 (Active Laundry & Dishwasher)",
            "REDD House 1 (Full 4-Appliance Coverage)",
        ],
        index=0,
    )

    h_id = 2 if "House 2" in house_choice else (3 if "House 3" in house_choice else 1)
    df_stream = load_house_stream_data(h_id)

    if df_stream.empty:
        st.error(f"Could not load data for House {h_id}. Ensure `data/processed/redd_real_house_{h_id}.csv` exists.")
        return

    mains_data = df_stream["mains"].values.astype(np.float32)
    timestamps = df_stream.index
    total_samples = len(mains_data)

    # 2. Preset Jump Points (for interesting events)
    st.sidebar.subheader("📍 Jump to Active Event")
    if h_id == 2:
        jump_options = {
            "Beginning (Hour 0)": 599,
            "Fridge Compressor Onset (Hour 6.5)": 3800,
            "Microwave & Dishwasher Bursts (Hour 14.5)": 8600,
            "Evening Load Cycles (Hour 24)": 14400,
        }
    elif h_id == 3:
        jump_options = {
            "Beginning (Hour 0)": 599,
            "Microwave Pulse (Hour 7.1)": 4200,
            "Dishwasher Run (Hour 34.4)": 20500,
            "High-Power Washing Machine Cycle (Hour 91.5)": 54800,
        }
    else:
        jump_options = {
            "Beginning (Hour 0)": 599,
            "Active Mealtime Bursts (Hour 12)": 7200,
            "Full Household Cycle (Hour 24)": 14400,
        }

    jump_choice = st.sidebar.selectbox("Event Bookmark", options=list(jump_options.keys()))

    # 3. Speed Control
    st.sidebar.subheader("⏩ Replay Speed")
    speed_presets = {
        "120x (1 hour in 30 sec)": (120, 2, 0.10),
        "360x (1 hour in 10 sec)": (360, 6, 0.10),
        "720x (1 hour in 5 sec - Benchmark)": (720, 12, 0.10),
        "1440x (1 hour in 2.5 sec)": (1440, 24, 0.10),
        "3600x (1 hour in 1 sec - Ultra fast)": (3600, 60, 0.10),
    }

    speed_name = st.sidebar.select_slider(
        "Playback Speed Ratio",
        options=list(speed_presets.keys()),
        value="720x (1 hour in 5 sec - Benchmark)",
    )
    speed_ratio, step_samples, sleep_sec = speed_presets[speed_name]

    # Additional options
    show_ground_truth = st.sidebar.checkbox("Overlay Submeter Ground Truth", value=True)
    history_window = st.sidebar.slider("Chart Rolling Display Window (samples)", 50, 300, 120, step=10)

    # --- Session State Initialization ---
    if "cursor" not in st.session_state:
        st.session_state.cursor = jump_options[jump_choice]
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "last_jump" not in st.session_state or st.session_state.last_jump != jump_choice:
        st.session_state.cursor = jump_options[jump_choice]
        st.session_state.last_jump = jump_choice
    if "cum_energy_kwh" not in st.session_state:
        st.session_state.cum_energy_kwh = 0.0

    # --- Transport Controls (Play / Pause / Reset / Step) ---
    col_play, col_pause, col_step, col_reset, col_jump = st.columns([1, 1, 1, 1, 2])
    with col_play:
        if st.button("▶️ Play / Resume", use_container_width=True):
            st.session_state.is_running = True
    with col_pause:
        if st.button("⏸️ Pause", use_container_width=True):
            st.session_state.is_running = False
    with col_step:
        if st.button("⏭️ Step (1 hr)", use_container_width=True):
            st.session_state.cursor = min(st.session_state.cursor + 600, total_samples - 1)
    with col_reset:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.cursor = jump_options[jump_choice]
            st.session_state.is_running = False
            st.session_state.cum_energy_kwh = 0.0

    # Stream progress slider
    progress_pct = float(st.session_state.cursor / total_samples)
    st.progress(progress_pct, text=f"Feed Progress: Sample {st.session_state.cursor:,} of {total_samples:,} ({(st.session_state.cursor * 6 / 3600):.1f} / {(total_samples * 6 / 3600):.1f} hours)")

    # Dynamic UI Placeholders
    top_kpi_placeholder = st.empty()
    badges_placeholder = st.empty()
    chart_placeholder = st.empty()

    # Pre-allocate buffer for rolling chart history
    if "history_mains" not in st.session_state:
        st.session_state.history_mains = []
        st.session_state.history_preds = {app: [] for app in appliances}
        st.session_state.history_trues = {app: [] for app in appliances}
        st.session_state.history_times = []

    # --- Live Simulation Loop ---
    while st.session_state.is_running and st.session_state.cursor < total_samples:
        curr_idx = st.session_state.cursor
        window_start = max(0, curr_idx - 599)
        window_mains = mains_data[window_start:curr_idx]

        if len(window_mains) < 599:
            # Left-pad with first value if at stream boundary
            window_mains = np.pad(window_mains, (599 - len(window_mains), 0), mode="edge")

        # Instantaneous readings
        curr_time = timestamps[curr_idx]
        curr_mains_w = float(mains_data[curr_idx])
        st.session_state.cum_energy_kwh += (curr_mains_w * (6.0 * step_samples) / 3600.0) / 1000.0

        # Run model inference on rolling 599-sample window
        m_norm = norm_params.normalize_mains(window_mains)
        x_tensor = torch.from_numpy(m_norm[np.newaxis, :, np.newaxis]).float()

        with torch.no_grad():
            preds = model(x_tensor)
            p_last = preds["power"][0, -1].numpy()
            o_last = preds["on_off"][0, -1].numpy()

        instant_watts: Dict[str, float] = {}
        instant_onoff: Dict[str, bool] = {}
        instant_true_watts: Dict[str, float] = {}

        for i, app in enumerate(appliances):
            w_pred = norm_params.denormalize_appliance(np.array([p_last[i]]), app)[0]
            w_pred = float(np.clip(w_pred, 0.0, None))
            thresh = DEFAULT_APPLIANCE_THRESHOLDS.get(app, 20.0)

            instant_watts[app] = w_pred
            # On/off decision based on probability or physical power
            instant_onoff[app] = bool(o_last[i] >= 0.35 or w_pred >= thresh)

            if app in df_stream.columns:
                instant_true_watts[app] = float(df_stream[app].iloc[curr_idx])
            else:
                instant_true_watts[app] = 0.0

        # Update History Buffer
        st.session_state.history_times.append(curr_time)
        st.session_state.history_mains.append(curr_mains_w)
        for app in appliances:
            st.session_state.history_preds[app].append(instant_watts[app])
            st.session_state.history_trues[app].append(instant_true_watts[app])

        # Trim history to display window size
        if len(st.session_state.history_times) > history_window:
            st.session_state.history_times = st.session_state.history_times[-history_window:]
            st.session_state.history_mains = st.session_state.history_mains[-history_window:]
            for app in appliances:
                st.session_state.history_preds[app] = st.session_state.history_preds[app][-history_window:]
                st.session_state.history_trues[app] = st.session_state.history_trues[app][-history_window:]

        # 1. Top KPI Row
        with top_kpi_placeholder.container():
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Instant Aggregate Mains", f"{curr_mains_w:.1f} W")
            k2.metric("Feed Timestamp", str(curr_time).split("+")[0])
            k3.metric("Replay Speed", f"{speed_name.split(' ')[0]}")
            k4.metric("Session Energy", f"{st.session_state.cum_energy_kwh:.3f} kWh")

        # 2. Appliance LED Status Cards
        with badges_placeholder.container():
            st.markdown("### 🔌 Disaggregated Appliance States (Real-Time)")
            card_cols = st.columns(len(appliances))
            for i, app in enumerate(appliances):
                is_on = instant_onoff[app]
                w = instant_watts[app]
                true_w = instant_true_watts[app]

                status_html = "<span class='badge-on'>🟢 ON</span>" if is_on else "<span class='badge-off'>⚪ OFF</span>"
                gt_text = f"Submeter: **{true_w:.1f} W**" if app in df_stream.columns else "Submeter: *Unmonitored*"

                with card_cols[i]:
                    st.markdown(
                        f"""
                        <div class="metric-box">
                            <h4 style="margin: 0;">{app.replace('_', ' ').title()}</h4>
                            <p style="margin: 6px 0;">{status_html}</p>
                            <h3 style="margin: 4px 0; color: #0077b6;">{w:.1f} W</h3>
                            <small style="color: #6c757d;">{gt_text}</small>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # 3. Dynamic Rolling Power Chart
        with chart_placeholder.container():
            st.markdown(f"### 📊 Rolling Disaggregation Window (Last {history_window * 6} seconds)")
            chart_df = pd.DataFrame(index=st.session_state.history_times)
            chart_df["Aggregate Mains (W)"] = st.session_state.history_mains

            for app in appliances:
                chart_df[f"{app.replace('_', ' ').title()} (Est. Watts)"] = st.session_state.history_preds[app]
                if show_ground_truth and app in df_stream.columns:
                    chart_df[f"{app.replace('_', ' ').title()} (True Watts)"] = st.session_state.history_trues[app]

            st.line_chart(chart_df)

        st.session_state.cursor += step_samples
        time.sleep(sleep_sec)

    # When paused or stream finishes, still render the current state
    if not st.session_state.is_running:
        curr_idx = min(st.session_state.cursor, total_samples - 1)
        curr_time = timestamps[curr_idx]
        curr_mains_w = float(mains_data[curr_idx])

        with top_kpi_placeholder.container():
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Current Mains Power", f"{curr_mains_w:.1f} W", "PAUSED")
            k2.metric("Stream Position", f"Sample {curr_idx:,}")
            k3.metric("Replay Speed", f"{speed_name.split(' ')[0]}")
            k4.metric("Session Energy", f"{st.session_state.cum_energy_kwh:.3f} kWh")

        with badges_placeholder.container():
            st.info("⏸️ **Streaming Paused**. Click **▶️ Play / Resume** above to start accelerated replay.")


if __name__ == "__main__":
    main()
