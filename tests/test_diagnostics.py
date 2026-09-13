"""Unit tests for src/diagnostics.py Phase 0 diagnostics."""

import numpy as np
import pytest
from src.diagnostics import compute_phase0_diagnostics, format_phase0_diagnostic_table


def test_compute_phase0_diagnostics_perfect():
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0])
    # Perfect predictions
    y_pred = np.array([0.05, 0.1, 0.9, 0.85, 0.02, 0.95, 0.1, 0.05])
    
    diag = compute_phase0_diagnostics(y_true, y_pred)
    assert diag["ap"] == pytest.approx(1.0, abs=1e-3)
    assert diag["oracle_f1"] == pytest.approx(1.0, abs=1e-3)
    assert diag["f1_at_50"] == pytest.approx(1.0, abs=1e-3)
    assert diag["ceiling_gap"] == pytest.approx(0.0, abs=1e-3)
    assert diag["active_timesteps"] == 3
    assert diag["total_timesteps"] == 8


def test_compute_phase0_diagnostics_with_mask():
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0])
    y_pred = np.array([0.05, 0.1, 0.9, 0.85, 0.02, 0.95, 0.1, 0.05])
    mask = np.array([1, 1, 1, 0, 0, 1, 1, 1])  # mask out one active sample
    
    diag = compute_phase0_diagnostics(y_true, y_pred, mask=mask)
    assert diag["total_timesteps"] == 6
    assert diag["active_timesteps"] == 2


def test_compute_phase0_diagnostics_empty_or_no_positives():
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0.1, 0.2, 0.05, 0.01])
    
    diag = compute_phase0_diagnostics(y_true, y_pred)
    assert np.isnan(diag["ap"])
    assert np.isnan(diag["oracle_f1"])
    assert diag["active_timesteps"] == 0


def test_format_phase0_diagnostic_table():
    diagnostics = {
        "fridge": {
            "prevalence": 0.25,
            "active_timesteps": 250,
            "total_timesteps": 1000,
            "ap": 0.85,
            "f1_at_50": 0.82,
            "oracle_f1": 0.86,
            "oracle_threshold": 0.42,
            "ceiling_gap": 0.04,
        }
    }
    df = format_phase0_diagnostic_table(diagnostics, house_id=1)
    assert len(df) == 1
    assert "House 1" in df.iloc[0]["House"]
    assert "fridge" in df.iloc[0]["Appliance"]
