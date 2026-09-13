"""Comprehensive unit tests for Phase 1 data pipeline masking and loss integration."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.data_pipeline import NormalizationParams, create_sliding_windows, NILMDataset
from src.loss import MultiApplianceLoss
from src.model import MultiApplianceNILM
from src.evaluate import evaluate_dataset


def test_create_sliding_windows_3d_mask():
    # Synthetic dataframe with 1000 timesteps
    length = 1000
    appliances = ["fridge", "microwave", "dishwasher", "washing_machine"]
    df = pd.DataFrame({
        "mains": np.random.uniform(100, 1000, length).astype(np.float32),
        "fridge": np.random.uniform(0, 200, length).astype(np.float32),
        "microwave": np.random.uniform(0, 1500, length).astype(np.float32),
        "dishwasher": np.full(length, np.nan),  # Dishwasher is all NaN
        "washing_machine": np.zeros(length, dtype=np.float32),
    })

    # House presence: microwave is unmetered
    presence = {
        "fridge": True,
        "microwave": False,
        "dishwasher": True,
        "washing_machine": True,
    }

    stats = {app: {"active_mean": 100.0, "active_std": 50.0, "threshold": 20.0} for app in appliances}
    norm_params = NormalizationParams(mains_mean=500.0, mains_std=200.0, appliance_stats=stats)

    x, yp, yo, mask = create_sliding_windows(
        df=df,
        appliances=appliances,
        norm_params=norm_params,
        window_length=200,
        stride=100,
        appliance_presence=presence,
    )

    assert len(x) > 0
    assert x.shape == (len(x), 200, 1)
    assert yp.shape == (len(x), 200, 4)
    assert yo.shape == (len(x), 200, 4)
    assert mask.shape == (len(x), 200, 4)

    # Fridge: present and non-NaN -> mask is 1.0
    assert np.all(mask[:, :, 0] == 1.0)

    # Microwave: unmetered -> mask is 0.0 everywhere
    assert np.all(mask[:, :, 1] == 0.0)

    # Dishwasher: all NaN -> mask is 0.0 everywhere
    assert np.all(mask[:, :, 2] == 0.0)

    # Washing machine: present, valid -> mask is 1.0
    assert np.all(mask[:, :, 3] == 1.0)


def test_masked_loss_zero_gradient_on_masked_channel():
    appliances = ["fridge", "microwave"]
    loss_fn = MultiApplianceLoss(appliances=appliances, lambda_bce=1.0)

    batch_size = 2
    length = 10
    num_apps = 2

    power_pred = torch.randn(batch_size, length, num_apps, requires_grad=True)
    raw_onoff = torch.randn(batch_size, length, num_apps, requires_grad=True)
    onoff_pred = torch.sigmoid(raw_onoff)
    power_true = torch.zeros(batch_size, length, num_apps)
    onoff_true = torch.zeros(batch_size, length, num_apps)

    # Appliance 0 (fridge) is valid (1.0), Appliance 1 (microwave) is masked out (0.0)
    app_mask = torch.zeros(batch_size, length, num_apps)
    app_mask[:, :, 0] = 1.0
    app_mask[:, :, 1] = 0.0

    loss, breakdown = loss_fn(power_pred, power_true, onoff_pred, onoff_true, appliance_mask=app_mask)

    assert breakdown["microwave_loss"] == 0.0
    assert breakdown["fridge_loss"] > 0.0

    loss.backward()

    # Gradients for appliance 1 (microwave) should be identically zero
    assert torch.all(power_pred.grad[:, :, 1] == 0.0)
    assert torch.all(raw_onoff.grad[:, :, 1] == 0.0)

    # Gradients for appliance 0 (fridge) should be non-zero
    assert torch.any(power_pred.grad[:, :, 0] != 0.0)
    assert torch.any(raw_onoff.grad[:, :, 0] != 0.0)


def test_evaluate_dataset_with_unmetered_appliance():
    appliances = ["fridge", "microwave"]
    stats = {app: {"active_mean": 100.0, "active_std": 50.0, "threshold": 20.0} for app in appliances}
    norm_params = NormalizationParams(mains_mean=500.0, mains_std=200.0, appliance_stats=stats)

    model = MultiApplianceNILM(appliances=appliances)

    N = 4
    W = 100
    x = np.random.randn(N, W, 1).astype(np.float32)
    yp = np.random.randn(N, W, 2).astype(np.float32)
    yo = np.random.randint(0, 2, (N, W, 2)).astype(np.float32)

    # Appliance 0 valid, Appliance 1 unmetered
    mask = np.zeros((N, W, 2), dtype=np.float32)
    mask[:, :, 0] = 1.0
    mask[:, :, 1] = 0.0

    ds = NILMDataset(x, yp, yo, mask)
    results = evaluate_dataset(model, ds, norm_params, appliances, device="cpu")

    # Fridge should have valid evaluation numbers
    assert "fridge" in results
    assert not np.isnan(results["fridge"]["f1"]) or results["fridge"]["active_samples"] == 0

    # Microwave is completely unmetered: should be NaN
    assert "microwave" in results
    assert np.isnan(results["microwave"]["f1"])
    assert np.isnan(results["microwave"]["nde"])
    assert np.isnan(results["microwave"]["ap"])
    assert results["microwave"]["valid_samples"] == 0
