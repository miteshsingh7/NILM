"""Streamlit Cost & Usage Attribution Dashboard for NILM.

Inputs:
- Trained model's per-appliance disaggregated power output for a given house/time window.
- Pre-cached 14-day disaggregated datasets for REDD House 2 and House 3, or live neural inference.

Outputs:
1. Configurable $/kWh tariff input (with multi-currency support: $, ₹, £, €).
2. Per-appliance cost breakdown (Daily and Weekly).
3. Stacked bar and area charts of usage by appliance over time.
4. On/Off event detection, duty cycles, and ground-truth validation comparison.
"""

from pathlib import Path
import sys
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
from src.utils import calculate_cost, calculate_energy_kwh, compute_nde


# Configure Streamlit Page
st.set_page_config(
    page_title="NILM Cost & Usage Attribution Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished UI
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #e9ecef;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 10px;
        padding-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_checkpoint(checkpoint_path: str = "checkpoints/best_model.pt"):
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
def load_demo_data(house_id: int) -> pd.DataFrame:
    """Loads pre-cached 14-day disaggregated demo data for House 2 or 3."""
    demo_file = ROOT_DIR / f"data/processed/disagg_demo_house_{house_id}.csv"
    if demo_file.exists():
        df = pd.read_csv(demo_file, index_col=0, parse_dates=True)
        return df

    # Fallback: load raw house CSV
    raw_file = ROOT_DIR / f"data/processed/redd_real_house_{house_id}.csv"
    if raw_file.exists():
        df = pd.read_csv(raw_file, index_col=0, parse_dates=True)
        return df
    return pd.DataFrame()


def run_live_disaggregation(
    model: MultiApplianceNILM,
    norm_params: NormalizationParams,
    mains_series: np.ndarray,
    appliances: List[str],
    window_length: int = 599,
) -> Tuple[np.ndarray, np.ndarray]:
    """Runs sliding-window inference on arbitrary input mains series."""
    total_len = len(mains_series)
    num_apps = len(appliances)
    power_accum = np.zeros((total_len, num_apps), dtype=np.float32)
    onoff_accum = np.zeros((total_len, num_apps), dtype=np.float32)
    counts = np.zeros(total_len, dtype=np.float32)

    stride = window_length // 4
    windows = []
    indices = []
    for start in range(0, total_len - window_length + 1, stride):
        end = start + window_length
        m_win = mains_series[start:end]
        windows.append(norm_params.normalize_mains(m_win)[:, np.newaxis])
        indices.append((start, end))

    if not windows:
        return np.zeros((total_len, num_apps), dtype=np.float32), np.zeros((total_len, num_apps), dtype=np.float32)

    x_tensor = torch.from_numpy(np.array(windows)).float()
    with torch.no_grad():
        preds = model(x_tensor)
        p_all = preds["power"].numpy()
        o_all = preds["on_off"].numpy()

    for idx, (start, end) in enumerate(indices):
        power_accum[start:end] += p_all[idx]
        onoff_accum[start:end] += o_all[idx]
        counts[start:end] += 1.0

    valid = counts > 0
    power_accum[valid] /= counts[valid, np.newaxis]
    onoff_accum[valid] /= counts[valid, np.newaxis]

    pred_watts = np.zeros_like(power_accum)
    for i, app in enumerate(appliances):
        pred_watts[:, i] = np.clip(norm_params.denormalize_appliance(power_accum[:, i], app), 0.0, None)

    return pred_watts, onoff_accum


def main():
    # --- Header ---
    st.title("⚡ Smart-Meter Disaggregation: Cost & Usage Attribution")
    st.markdown(
        "**Multi-Target Seq2Seq Framework** | Disaggregates smart-meter aggregate power into individual appliance "
        "consumption profiles, attributes energy & electricity bill costs, and tracks daily/weekly load signatures."
    )
    st.markdown("---")

    # --- Sidebar Controls ---
    st.sidebar.header("🛠️ Dashboard Configuration")

    # 1. Tariff Configuration
    st.sidebar.subheader("💵 Electricity Tariff")
    currency = st.sidebar.selectbox("Currency Symbol", options=["$", "₹", "£", "€"], index=0)
    tariff = st.sidebar.number_input(
        f"Tariff ({currency}/kWh)",
        min_value=0.01,
        max_value=2.00 if currency != "₹" else 50.00,
        value=0.15 if currency != "₹" else 7.50,
        step=0.01 if currency != "₹" else 0.25,
        help="Electricity supply rate per kilowatt-hour used to calculate daily and weekly financial attribution.",
    )

    # 2. Data Source Selection
    st.sidebar.subheader("📂 Data Source")
    data_source = st.sidebar.radio(
        "Select Source",
        options=[
            "REDD House 2 (Held-Out Generalization — 14 Days)",
            "REDD House 3 (High Load & Active Washer — 14 Days)",
            "Live Model Inference (Full REDD House CSV)",
            "Upload Custom CSV",
        ],
        index=0,
    )

    appliances = DEFAULT_APPLIANCES
    df_raw = pd.DataFrame()

    if "House 2" in data_source:
        df_raw = load_demo_data(2)
        st.sidebar.caption("🏠 **House 2**: 14 days (201,600 samples @ 6s). Completely unseen during training.")
    elif "House 3" in data_source:
        df_raw = load_demo_data(3)
        st.sidebar.caption("🏠 **House 3**: 14 days (201,600 samples @ 6s). Features frequent washing machine runs.")
    elif "Live Model Inference" in data_source:
        house_num = st.sidebar.selectbox("Select REDD House", options=[1, 2, 3, 4, 5, 6], index=1)
        csv_file = ROOT_DIR / f"data/processed/redd_real_house_{house_num}.csv"
        if csv_file.exists():
            df_full = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            # Default to first 7 days for responsive interactive inference
            days_to_load = st.sidebar.slider("Days to disaggregate", min_value=1, max_value=14, value=7)
            df_raw = df_full.iloc[: int(days_to_load * 86400 / 6)].copy()
        else:
            st.sidebar.error(f"File {csv_file} not found.")
    elif "Upload Custom CSV" in data_source:
        uploaded_file = st.sidebar.file_uploader("Upload CSV (datetime index, 'mains' column)", type=["csv"])
        if uploaded_file is not None:
            df_raw = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)

    if df_raw.empty:
        st.warning("⚠️ No data loaded. Please verify data files or select another source.")
        return

    # Ensure disaggregated columns exist
    has_precomputed = any(f"{app}_pred_watts" in df_raw.columns for app in appliances)

    if not has_precomputed:
        model, norm_params, ckpt_apps = load_checkpoint("checkpoints/best_model.pt")
        if model is None:
            st.error("Model checkpoint not found. Please train model or run cache script.")
            return
        with st.spinner("Running deep neural network disaggregation inference..."):
            mains_vals = df_raw["mains"].values.astype(np.float32)
            pred_watts, pred_onoff = run_live_disaggregation(
                model=model,
                norm_params=norm_params,
                mains_series=mains_vals,
                appliances=appliances,
            )
            for i, app in enumerate(appliances):
                df_raw[f"{app}_pred_watts"] = pred_watts[:, i]
                df_raw[f"{app}_pred_prob"] = pred_onoff[:, i]

    # --- Time Window Selector ---
    st.sidebar.subheader("📅 Time Window Filter")
    min_date = df_raw.index.min().to_pydatetime()
    max_date = df_raw.index.max().to_pydatetime()

    preset_window = st.sidebar.selectbox(
        "Window Preset",
        options=["Full Duration", "Last 7 Days (Weekly)", "Last 24 Hours (Daily)", "Custom Range"],
        index=1,
    )

    if preset_window == "Last 24 Hours (Daily)":
        start_date = max_date - pd.Timedelta(hours=24)
        end_date = max_date
    elif preset_window == "Last 7 Days (Weekly)":
        start_date = max_date - pd.Timedelta(days=7)
        end_date = max_date
    elif preset_window == "Custom Range":
        selected_range = st.sidebar.slider(
            "Select Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD HH:mm",
        )
        start_date, end_date = selected_range
    else:
        start_date, end_date = min_date, max_date

    df_slice = df_raw.loc[start_date:end_date].copy()

    if len(df_slice) < 10:
        st.warning("Selected window contains insufficient data points. Please expand the window.")
        return

    duration_hours = (df_slice.index.max() - df_slice.index.min()).total_seconds() / 3600.0
    duration_days = max(duration_hours / 24.0, 1.0 / 24.0)

    # --- Core Energy & Cost Math ---
    step_seconds = 6
    mains_kwh = calculate_energy_kwh(df_slice["mains"].values, step_seconds)
    mains_cost = calculate_cost(mains_kwh, tariff)

    app_kwh: Dict[str, float] = {}
    app_costs: Dict[str, float] = {}
    app_pred_w: Dict[str, np.ndarray] = {}

    for app in appliances:
        col = f"{app}_pred_watts"
        p_w = df_slice[col].values if col in df_slice.columns else np.zeros(len(df_slice))
        app_pred_w[app] = p_w
        kwh = calculate_energy_kwh(p_w, step_seconds)
        app_kwh[app] = kwh
        app_costs[app] = calculate_cost(kwh, tariff)

    total_disagg_kwh = sum(app_kwh.values())
    total_disagg_cost = sum(app_costs.values())
    unmetered_kwh = max(0.0, mains_kwh - total_disagg_kwh)
    unmetered_cost = calculate_cost(unmetered_kwh, tariff)

    # Daily and weekly scaling multipliers
    daily_factor = 1.0 / duration_days
    weekly_factor = 7.0 / duration_days

    # --- Top KPI Metric Row ---
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Selected Duration", f"{duration_days:.1f} Days", f"{len(df_slice):,} samples")
    kpi2.metric("Total Mains Energy", f"{mains_kwh:.2f} kWh", f"{currency}{mains_cost:.2f} Total")
    kpi3.metric(
        "Disaggregated Power",
        f"{total_disagg_kwh:.2f} kWh",
        f"{(total_disagg_kwh / mains_kwh * 100):.1f}% identified" if mains_kwh > 0 else "N/A",
    )
    kpi4.metric("Avg Daily Cost", f"{currency}{total_disagg_cost * daily_factor:.2f} / day")
    kpi5.metric("Projected Weekly Cost", f"{currency}{total_disagg_cost * weekly_factor:.2f} / week")

    st.markdown("---")

    # --- Primary Tabs ---
    tab_cost, tab_charts, tab_timeseries, tab_validation = st.tabs([
        "💰 Cost Attribution (Daily & Weekly)",
        "📊 Stacked Usage Charts",
        "📈 Time-Series Disaggregation",
        "🎯 Model Accuracy vs Ground Truth",
    ])

    # =========================================================================
    # TAB 1: PER-APPLIANCE COST BREAKDOWN (DAILY & WEEKLY)
    # =========================================================================
    with tab_cost:
        st.subheader("Per-Appliance Cost & Energy Attribution Breakdown")
        st.caption(
            f"Calculated at **{currency}{tariff:.2f} / kWh** tariff rate over the selected **{duration_days:.2f}-day** window."
        )

        cost_table_rows = []
        for app in appliances:
            kwh = app_kwh[app]
            cost = app_costs[app]
            pct = (kwh / mains_kwh * 100) if mains_kwh > 0 else 0.0

            cost_table_rows.append({
                "Appliance": app.replace("_", " ").title(),
                "Energy (kWh)": round(kwh, 3),
                "Share (%)": round(pct, 1),
                "Window Cost": round(cost, 2),
                "Daily Energy (kWh/d)": round(kwh * daily_factor, 3),
                "Daily Cost": round(cost * daily_factor, 2),
                "Weekly Energy (kWh/wk)": round(kwh * weekly_factor, 3),
                "Weekly Cost": round(cost * weekly_factor, 2),
            })

        # Unmetered / other loads
        cost_table_rows.append({
            "Appliance": "Other / Unmetered Baseline",
            "Energy (kWh)": round(unmetered_kwh, 3),
            "Share (%)": round((unmetered_kwh / mains_kwh * 100), 1) if mains_kwh > 0 else 0.0,
            "Window Cost": round(unmetered_cost, 2),
            "Daily Energy (kWh/d)": round(unmetered_kwh * daily_factor, 3),
            "Daily Cost": round(unmetered_cost * daily_factor, 2),
            "Weekly Energy (kWh/wk)": round(unmetered_kwh * weekly_factor, 3),
            "Weekly Cost": round(unmetered_cost * weekly_factor, 2),
        })

        summary_df = pd.DataFrame(cost_table_rows)

        # Display formatted table
        formatted_df = summary_df.copy()
        formatted_df["Window Cost"] = formatted_df["Window Cost"].apply(lambda x: f"{currency}{x:.2f}")
        formatted_df["Daily Cost"] = formatted_df["Daily Cost"].apply(lambda x: f"{currency}{x:.2f}")
        formatted_df["Weekly Cost"] = formatted_df["Weekly Cost"].apply(lambda x: f"{currency}{x:.2f}")
        formatted_df["Share (%)"] = formatted_df["Share (%)"].apply(lambda x: f"{x:.1f}%")

        st.dataframe(formatted_df, use_container_width=True, hide_index=True)

        # Columns for side-by-side Daily vs Weekly charts
        col_d, col_w = st.columns(2)
        with col_d:
            st.markdown("#### 📅 Daily Cost Attribution per Appliance")
            daily_chart_df = summary_df.iloc[:-1].set_index("Appliance")[["Daily Cost"]]
            st.bar_chart(daily_chart_df)

        with col_w:
            st.markdown("#### 🗓️ Projected Weekly Cost Breakdown")
            weekly_chart_df = summary_df.iloc[:-1].set_index("Appliance")[["Weekly Cost"]]
            st.bar_chart(weekly_chart_df)

    # =========================================================================
    # TAB 2: STACKED BAR OR AREA CHART OF USAGE OVER TIME
    # =========================================================================
    with tab_charts:
        st.subheader("Appliance Consumption Share & Stacked Usage Over Time")

        chart_mode = st.radio(
            "Visualization Format",
            options=["Stacked Area Chart (Power over Time)", "Stacked Bar Chart (Daily Energy Usage)"],
            horizontal=True,
        )

        app_cols_display = [f"{app.replace('_', ' ').title()}" for app in appliances]
        power_df = pd.DataFrame(index=df_slice.index)
        for app in appliances:
            power_df[app.replace("_", " ").title()] = df_slice[f"{app}_pred_watts"]

        if chart_mode == "Stacked Area Chart (Power over Time)":
            st.markdown("#### Cumulative Stacked Power Profile (Watts)")
            st.caption("Shows continuous power contribution of each appliance superimposed across the timeline.")
            # Resample for smooth web rendering if window is large
            if len(power_df) > 10000:
                resampled_power = power_df.resample("1min").mean()
            else:
                resampled_power = power_df
            st.area_chart(resampled_power)

        else:
            st.markdown("#### Daily Energy Consumption by Appliance (kWh)")
            st.caption("Aggregated kilowatt-hours per day broken down by appliance.")

            # Compute daily kWh per appliance
            daily_energy_df = pd.DataFrame(index=df_slice.resample("1D").indices.keys())
            for app in appliances:
                # 6s sampling: sum(watts) * (6/3600) / 1000 = kWh
                daily_kwh = df_slice[f"{app}_pred_watts"].resample("1D").apply(lambda s: calculate_energy_kwh(s.values, 6))
                daily_energy_df[app.replace("_", " ").title()] = daily_kwh

            daily_energy_df.index = [d.strftime("%Y-%m-%d") for d in daily_energy_df.index]
            st.bar_chart(daily_energy_df)

            st.markdown(f"#### Daily Cost Attribution ({currency})")
            daily_cost_df = daily_energy_df * tariff
            st.bar_chart(daily_cost_df)

    # =========================================================================
    # TAB 3: TIME-SERIES DISAGGREGATION
    # =========================================================================
    with tab_timeseries:
        st.subheader("Instantaneous Smart-Meter Disaggregation Feed")
        st.caption("Compare aggregate mains power against disaggregated appliance outputs.")

        plot_df = pd.DataFrame(index=df_slice.index)
        plot_df["Mains Aggregate"] = df_slice["mains"]
        for app in appliances:
            plot_df[app.replace("_", " ").title()] = df_slice[f"{app}_pred_watts"]

        # If data is large, resample to 30s or 1min for responsive rendering
        if len(plot_df) > 5000:
            plot_resampled = plot_df.resample("30s").mean()
        else:
            plot_resampled = plot_df

        st.line_chart(plot_resampled)

        # On/Off state detection summary table
        st.markdown("#### Detected Events & Operational Summary")
        event_rows = []
        for app in appliances:
            pw = df_slice[f"{app}_pred_watts"].values
            thresh = DEFAULT_APPLIANCE_THRESHOLDS.get(app, 20.0)
            # Use standalone event detection logic
            ev_metrics = evaluate_appliance_events(
                y_pred=pw,
                y_true=pw,  # self-profile
                appliance=app,
                power_threshold=thresh,
            )
            active_hours = (ev_metrics["active_samples_pred"] * 6) / 3600.0
            duty_cycle = (ev_metrics["active_samples_pred"] / len(pw) * 100) if len(pw) > 0 else 0.0

            event_rows.append({
                "Appliance": app.replace("_", " ").title(),
                "Power Threshold": f"{thresh:.1f} W",
                "Peak Power (W)": f"{np.max(pw):.1f} W",
                "Average Power (W)": f"{np.mean(pw):.1f} W",
                "Active Hours": f"{active_hours:.2f} hrs",
                "Duty Cycle": f"{duty_cycle:.2f}%",
                "Status": ev_metrics["status_flag"],
            })

        st.dataframe(pd.DataFrame(event_rows), use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 4: GROUND TRUTH VALIDATION (HOUSE 2 / HOUSE 3)
    # =========================================================================
    with tab_validation:
        st.subheader("Disaggregation Accuracy vs Sub-metered Ground Truth")
        has_gt = any(f"{app}_true_watts" in df_slice.columns or app in df_slice.columns for app in appliances)

        if not has_gt:
            st.info("Submetered ground truth columns are not present in the current dataset.")
        else:
            val_rows = []
            for app in appliances:
                true_col = f"{app}_true_watts" if f"{app}_true_watts" in df_slice.columns else app
                pred_col = f"{app}_pred_watts"
                if true_col in df_slice.columns and pred_col in df_slice.columns:
                    y_t = df_slice[true_col].values
                    y_p = df_slice[pred_col].values
                    thresh = DEFAULT_APPLIANCE_THRESHOLDS.get(app, 20.0)
                    active_true = int(np.sum(y_t >= thresh))

                    if active_true > 0:
                        nde = compute_nde(y_t, y_p)
                        zero_nde = 1.0
                        beats_zero = "YES" if nde < zero_nde else "No"
                        nde_str = f"{nde:.4f}"
                    else:
                        nde_str = "NaN (0 active events)"
                        beats_zero = "N/A"

                    mae = float(np.mean(np.abs(y_p - y_t)))
                    ev = evaluate_appliance_events(y_p, y_t, appliance=app, power_threshold=thresh)

                    val_rows.append({
                        "Appliance": app.replace("_", " ").title(),
                        "Ground Truth kWh": round(calculate_energy_kwh(y_t, 6), 3),
                        "Disaggregated kWh": round(calculate_energy_kwh(y_p, 6), 3),
                        "Model NDE": nde_str,
                        "Beats Baseline?": beats_zero,
                        "F1 Score": f"{ev['f1']:.4f}",
                        "Precision": f"{ev['precision'] * 100:.1f}%",
                        "Recall": f"{ev['recall'] * 100:.1f}%",
                        "MAE (W)": f"{mae:.2f} W",
                        "Status": ev["status_flag"],
                    })

            st.dataframe(pd.DataFrame(val_rows), use_container_width=True, hide_index=True)

    # --- CSV Export Section ---
    st.markdown("---")
    st.subheader("📥 Export Attribution Data")
    csv_bytes = df_slice.to_csv().encode("utf-8")
    st.download_button(
        label="Download Window Disaggregation & Attribution CSV",
        data=csv_bytes,
        file_name=f"nilm_attribution_{preset_window.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
