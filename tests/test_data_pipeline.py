"""Unit tests for NILM data pipeline, normalization, labels.dat parsing, and window generation."""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch

from src.data_pipeline import (
    NormalizationParams,
    compute_normalization_params,
    create_sliding_windows,
    parse_labels_dat,
    resample_and_clean,
    NILMDataset,
)
from src.sample_data import generate_synthetic_house


def test_resample_and_clean():
    idx = pd.date_range("2026-01-01 00:00:00", periods=60, freq="1s")
    df = pd.DataFrame({"mains": np.ones(60) * 100.0, "fridge": np.ones(60) * 50.0}, index=idx)

    resampled = resample_and_clean(df, sample_period_seconds=6, max_gap_fill_samples=3)
    assert len(resampled) == 10
    assert np.allclose(resampled["mains"].values, 100.0)
    assert np.allclose(resampled["fridge"].values, 50.0)


def test_parse_labels_dat_with_typos():
    with tempfile.TemporaryDirectory() as tmp_dir:
        labels_file = Path(tmp_dir) / "labels.dat"
        with open(labels_file, "w") as f:
            f.write("1 mains\n")
            f.write("2 mains\n")
            f.write("5 refrigerator\n")
            f.write("6 dishwaser\n")  # REDD typo
            f.write("10 washer_dryer\n")
            f.write("11 microwave\n")
            f.write("20 washer_dryer\n")

        mapping = parse_labels_dat(tmp_dir)
        assert mapping["mains"] == [1, 2]
        assert mapping["fridge"] == [5]
        assert mapping["dishwasher"] == [6]
        assert mapping["microwave"] == [11]
        assert mapping["washing_machine"] == [10, 20]


def test_active_period_normalization():
    np.random.seed(42)
    fridge_power = np.zeros(1000)
    fridge_power[100:300] = np.random.normal(150.0, 10.0, 200)

    mains_power = fridge_power + 100.0

    df = pd.DataFrame({"mains": mains_power, "fridge": fridge_power})
    norm_params = compute_normalization_params(df, appliances=["fridge"], thresholds={"fridge": 50.0})

    active_mean = norm_params.appliance_stats["fridge"]["active_mean"]
    assert 140.0 < active_mean < 160.0

    norm_val = norm_params.normalize_appliance(np.array([150.0]), "fridge")
    recovered = norm_params.denormalize_appliance(norm_val, "fridge")
    assert np.isclose(recovered[0], 150.0, atol=1e-3)


def test_sliding_window_shapes():
    df = generate_synthetic_house(num_days=2, sample_period_seconds=6, seed=42)
    appliances = ["fridge", "microwave", "dishwasher", "washing_machine"]
    norm_params = compute_normalization_params(df, appliances=appliances)

    window_len = 599
    stride = 149

    x, y_p, y_o, m = create_sliding_windows(
        df,
        appliances=appliances,
        norm_params=norm_params,
        window_length=window_len,
        stride=stride,
        appliance_presence={"fridge": True, "microwave": True, "dishwasher": True, "washing_machine": True},
    )

    assert x.ndim == 3
    assert x.shape[1] == 599
    assert x.shape[2] == 1  # Mains channel

    assert y_p.ndim == 3
    assert y_p.shape[1] == 599
    assert y_p.shape[2] == len(appliances)

    assert y_o.ndim == 3
    assert y_o.shape[1] == 599
    assert y_o.shape[2] == len(appliances)

    assert m.ndim == 3
    assert m.shape[1] == 599
    assert m.shape[2] == len(appliances)

    # Check dataset
    dataset = NILMDataset(x, y_p, y_o, m)
    assert len(dataset) == len(x)
    x_item, yp_item, yo_item, m_item = dataset[0]
    assert isinstance(x_item, torch.Tensor)
    assert x_item.shape == (599, 1)
    assert m_item.shape == (599, len(appliances))


def test_multiscale_channels_and_windows():
    from src.data_pipeline import construct_multiscale_mains_channels

    mains_1d = np.array([100.0, 110.0, 105.0, 120.0, 150.0, 140.0], dtype=np.float32)
    channels = construct_multiscale_mains_channels(mains_1d, window=3)
    assert channels.shape == (6, 3)
    assert not np.isnan(channels).any()
    # Check that channel 0 is identical to input
    assert np.allclose(channels[:, 0], mains_1d)

    # Test sliding windows with in_channels=3
    df = generate_synthetic_house(num_days=2, sample_period_seconds=6, seed=42)
    appliances = ["fridge", "microwave", "dishwasher", "washing_machine"]
    norm_params = compute_normalization_params(df, appliances=appliances)

    x, yp, yo, m = create_sliding_windows(
        df,
        appliances=appliances,
        norm_params=norm_params,
        window_length=599,
        stride=149,
        in_channels=3,
    )
    assert x.ndim == 3
    assert x.shape[1] == 599
    assert x.shape[2] == 3
    assert not np.isnan(x).any()

