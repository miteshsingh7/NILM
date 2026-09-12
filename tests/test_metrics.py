"""Unit tests for NILM evaluation metrics, energy integration, and tariff cost calculations."""

import numpy as np
import pytest

from src.utils import (
    calculate_cost,
    calculate_energy_kwh,
    compute_f1_score,
    compute_mae,
    compute_nde,
    compute_sae,
)


def test_calculate_energy_kwh():
    # 1000 Watts for 1 hour (600 samples at 6s)
    # Total watt-hours = 1000 Wh = 1.0 kWh
    watts = np.ones(600) * 1000.0
    kwh = calculate_energy_kwh(watts, step_seconds=6)
    assert np.isclose(kwh, 1.0, atol=1e-5)


def test_calculate_cost():
    # 2.5 kWh at ₹8.00 / kWh = ₹20.00
    cost = calculate_cost(2.5, 8.0)
    assert np.isclose(cost, 20.0, atol=1e-5)


def test_compute_nde():
    y_true = np.array([100.0, 200.0, 300.0])
    # Perfect prediction: NDE = 0.0
    nde_zero = compute_nde(y_true, y_true)
    assert np.isclose(nde_zero, 0.0, atol=1e-6)

    # Completely off prediction
    y_pred = np.array([110.0, 190.0, 310.0])
    nde = compute_nde(y_true, y_pred)
    # Expected: sqrt((100+100+100) / (10000+40000+90000)) = sqrt(300 / 140000) = ~0.04629
    expected_nde = np.sqrt(300.0 / 140000.0)
    assert np.isclose(nde, expected_nde, atol=1e-4)


def test_compute_f1_score():
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred_prob = np.array([0.9, 0.8, 0.1, 0.7, 0.2])  # 2 TP, 1 FP, 1 FN, 1 TN
    metrics = compute_f1_score(y_true, y_pred_prob, threshold=0.5)

    # TP = 2, FP = 1 -> Precision = 2/3
    # FN = 1 -> Recall = 2/3
    # F1 = 2/3
    assert np.isclose(metrics["precision"], 2.0 / 3.0, atol=1e-4)
    assert np.isclose(metrics["recall"], 2.0 / 3.0, atol=1e-4)
    assert np.isclose(metrics["f1"], 2.0 / 3.0, atol=1e-4)


def test_compute_sae():
    y_true = np.ones(600) * 1000.0  # 1.0 kWh
    y_pred = np.ones(600) * 800.0   # 0.8 kWh
    sae = compute_sae(y_true, y_pred, step_seconds=6)
    # SAE = |0.8 - 1.0| / 1.0 = 0.2
    assert np.isclose(sae, 0.2, atol=1e-4)
