"""Unit tests for standalone event detection module."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.config import DEFAULT_APPLIANCE_THRESHOLDS
from src.event_detection import (
    compute_event_metrics,
    detect_appliance_state,
    evaluate_appliance_events,
    evaluate_disaggregation_events,
    format_metrics_table,
)
from src.utils import compute_f1_score


def test_scoring_parity_with_compute_f1_score():
    """Confirms exact score parity with the legacy compute_f1_score from Prompts 1 & 2."""
    np.random.seed(123)
    y_true_binary = np.random.choice([0, 1], size=1000, p=[0.8, 0.2])
    y_pred_prob = np.random.uniform(0.0, 1.0, size=1000)

    # Legacy scoring
    legacy = compute_f1_score(y_true_binary, y_pred_prob, threshold=0.5)

    # New event detection scoring with probabilities
    new_metrics = evaluate_appliance_events(
        y_pred=y_pred_prob,
        y_true=y_true_binary,
        appliance="fridge",
        decision_threshold=0.5,
    )

    assert np.isclose(new_metrics["precision"], legacy["precision"], atol=1e-4)
    assert np.isclose(new_metrics["recall"], legacy["recall"], atol=1e-4)
    assert np.isclose(new_metrics["f1"], legacy["f1"], atol=1e-4)
    assert np.isclose(new_metrics["accuracy"], legacy["accuracy"], atol=1e-4)


def test_continuous_power_binarization():
    """Confirms that continuous power in Watts is properly binarized at threshold."""
    # 5 samples: [10W, 45W, 50W, 120W, 0W] -> threshold 50W -> [0, 0, 1, 1, 0]
    p_true = np.array([10.0, 45.0, 50.0, 120.0, 0.0])
    p_pred = np.array([5.0, 60.0, 55.0, 110.0, 0.0])  # -> [0, 1, 1, 1, 0]
    # TP: 2 (idx 2, 3), FP: 1 (idx 1), FN: 0, TN: 2 (idx 0, 4)
    # Precision: 2/3, Recall: 2/2 = 1.0, F1: 2 * (2/3) * 1 / (5/3) = 0.80

    res = evaluate_appliance_events(
        y_pred=p_pred,
        y_true=p_true,
        appliance="fridge",
        power_threshold=50.0,
    )

    assert res["tp"] == 2
    assert res["fp"] == 1
    assert res["fn"] == 0
    assert res["tn"] == 2
    assert np.isclose(res["precision"], 2.0 / 3.0, atol=1e-4)
    assert np.isclose(res["recall"], 1.0, atol=1e-4)
    assert np.isclose(res["f1"], 0.8, atol=1e-4)


def test_pandas_and_tensor_compatibility():
    """Confirms that PyTorch tensors and Pandas Series work seamlessly."""
    t_series = pd.Series([0.0, 100.0, 200.0, 0.0, 150.0])
    p_tensor = torch.tensor([0.0, 95.0, 210.0, 30.0, 0.0], dtype=torch.float32)

    res = evaluate_appliance_events(
        y_pred=p_tensor,
        y_true=t_series,
        appliance="microwave",
        power_threshold=100.0,
    )

    assert isinstance(res["f1"], float)
    assert res["total_samples"] == 5


def test_status_flagging_unmonitored_and_near_zero():
    """Confirms proper status flagging for unmonitored or rare-event appliances."""
    # 1. Unmonitored
    y_zero = np.zeros(500)
    res_unmonitored = evaluate_appliance_events(y_zero, y_zero, appliance="washing_machine")
    assert "UNMONITORED" in res_unmonitored["status_flag"]

    # 2. Near-zero activity (50 active samples out of 1000)
    y_near_zero = np.zeros(1000)
    y_near_zero[:50] = 100.0
    res_near_zero = evaluate_appliance_events(y_near_zero, y_near_zero, appliance="dishwasher")
    assert "NEAR-ZERO" in res_near_zero["status_flag"]

    # 3. Active (500 active samples)
    y_active = np.zeros(1000)
    y_active[:500] = 100.0
    res_active = evaluate_appliance_events(y_active, y_active, appliance="fridge")
    assert res_active["status_flag"] == "ACTIVE"


def test_multi_appliance_dataframe_evaluation():
    """Confirms evaluate_disaggregation_events handles multiple appliances in DataFrames."""
    df_true = pd.DataFrame({
        "fridge": [0.0, 60.0, 80.0, 0.0],
        "microwave": [0.0, 0.0, 1200.0, 0.0],
    })
    df_pred = pd.DataFrame({
        "fridge": [0.0, 70.0, 75.0, 0.0],
        "microwave": [0.0, 0.0, 1150.0, 0.0],
    })

    results = evaluate_disaggregation_events(df_pred, df_true)
    assert "fridge" in results
    assert "microwave" in results
    assert results["fridge"]["f1"] == 1.0
    assert results["microwave"]["f1"] == 1.0

    table = format_metrics_table(results)
    assert len(table) == 2
