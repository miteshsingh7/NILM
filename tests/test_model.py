"""Unit tests for the PyTorch shared encoder and multi-appliance head architecture."""

import pytest
import torch

from src.loss import MultiApplianceLoss
from src.model import ApplianceHead, MultiApplianceNILM, SharedEncoder, DecoupledTemporalNILM


def test_shared_encoder_forward():
    encoder = SharedEncoder(in_channels=1, lstm_hidden=128)
    batch_size = 4
    seq_len = 599

    x = torch.randn(batch_size, seq_len, 1)
    out = encoder(x)
    assert out.shape == (batch_size, seq_len, 256)

    x_transposed = torch.randn(batch_size, 1, seq_len)
    out_t = encoder(x_transposed)
    assert out_t.shape == (batch_size, seq_len, 256)


def test_appliance_head_forward():
    head = ApplianceHead(in_features=256, conv_filters=64, dense_dim=32)
    batch_size = 4
    seq_len = 599
    features = torch.randn(batch_size, seq_len, 256)

    power, onoff = head(features)
    assert power.shape == (batch_size, seq_len, 1)
    assert onoff.shape == (batch_size, seq_len, 1)
    assert torch.all(onoff >= 0.0) and torch.all(onoff <= 1.0)


def test_multi_appliance_model_end_to_end_with_on_weight():
    appliances = ["fridge", "microwave", "dishwasher", "washing_machine"]
    model = MultiApplianceNILM(appliances=appliances)
    criterion = MultiApplianceLoss(appliances=appliances, lambda_bce=1.0, on_weight=8.0)

    batch_size = 2
    seq_len = 599

    x = torch.randn(batch_size, seq_len, 1)
    y_power = torch.randn(batch_size, seq_len, len(appliances))
    y_onoff = torch.randint(0, 2, (batch_size, seq_len, len(appliances))).float()
    mask = torch.ones((batch_size, len(appliances)))

    preds = model(x)

    assert preds["power"].shape == (batch_size, seq_len, 4)
    assert preds["on_off"].shape == (batch_size, seq_len, 4)

    for app in appliances:
        assert app in preds["power_dict"]
        assert preds["power_dict"][app].shape == (batch_size, seq_len, 1)

    loss, breakdown = criterion(
        power_pred=preds["power"],
        power_true=y_power,
        onoff_pred=preds["on_off"],
        onoff_true=y_onoff,
        appliance_mask=mask,
    )
    assert loss.item() > 0
    assert "fridge_mse" in breakdown
    assert "fridge_bce" in breakdown

    loss.backward()

    for param in model.encoder.parameters():
        assert param.grad is not None
        break

    for app in appliances:
        for param in model.heads[app].parameters():
            assert param.grad is not None
            break


def test_on_weighting_increases_loss_on_active_burst():
    # If error is only during active periods, on_weight=8 should produce higher loss than on_weight=1
    appliances = ["microwave"]
    crit_unweighted = MultiApplianceLoss(appliances=appliances, on_weight=1.0, lambda_bce=0.0)
    crit_weighted = MultiApplianceLoss(appliances=appliances, on_weight=8.0, lambda_bce=0.0)

    p_pred = torch.zeros(1, 100, 1)
    p_true = torch.zeros(1, 100, 1)
    # 5 active timesteps with error
    p_true[0, 50:55, 0] = 5.0
    o_true = torch.zeros(1, 100, 1)
    o_true[0, 50:55, 0] = 1.0

    loss_unw, _ = crit_unweighted(p_pred, p_true, o_true, o_true)
    loss_w, _ = crit_weighted(p_pred, p_true, o_true, o_true)

    # In weighted loss, active steps have weight 8x
    assert loss_w.item() > loss_unw.item()


def test_normalization_types_forward():
    """Verify GroupNorm and LayerNorm work seamlessly in SharedEncoder and MultiApplianceNILM."""
    x = torch.randn(2, 100, 1)
    appliances = ["fridge", "microwave"]

    for norm in ["batchnorm", "groupnorm", "layernorm"]:
        encoder = SharedEncoder(in_channels=1, norm_type=norm, lstm_hidden=32)
        out = encoder(x)
        assert out.shape == (2, 100, 64)

        model = MultiApplianceNILM(appliances=appliances, norm_type=norm, lstm_hidden=32)
        preds = model(x)
        assert preds["power"].shape == (2, 100, 2)
        assert preds["on_off"].shape == (2, 100, 2)


def test_decoupled_temporal_nilm():
    """Verify DecoupledTemporalNILM forward pass, shapes, and decoupled gradients."""
    appliances = ["fridge", "microwave", "dishwasher", "washing_machine"]
    model = DecoupledTemporalNILM(
        appliances=appliances,
        in_channels=1,
        lstm_hidden=48,  # Parameter-matched
        norm_type="groupnorm",
    )

    batch_size = 2
    seq_len = 200
    x = torch.randn(batch_size, seq_len, 1)

    preds = model(x)

    assert preds["power"].shape == (batch_size, seq_len, 4)
    assert preds["on_off"].shape == (batch_size, seq_len, 4)

    # Verify per-appliance LSTM parameters exist and are decoupled
    for app in appliances:
        assert app in model.lstms
        assert app in model.heads
        assert preds["power_dict"][app].shape == (batch_size, seq_len, 1)

    # Verify backpropagation through a single appliance head only affects its own BiLSTM
    fridge_loss = preds["power_dict"]["fridge"].sum()
    fridge_loss.backward()

    # Fridge LSTM should have gradients
    for p in model.lstms["fridge"].parameters():
        assert p.grad is not None
        break

    # Microwave LSTM should NOT have gradients from fridge loss
    for p in model.lstms["microwave"].parameters():
        assert p.grad is None


def test_multiscale_and_transition_heads():
    appliances = ["fridge", "microwave", "dishwasher", "washing_machine"]
    model = DecoupledTemporalNILM(
        appliances=appliances,
        in_channels=3,
        lstm_hidden=32,
        norm_type="groupnorm",
        predict_transitions=True,
    )

    batch_size = 2
    seq_len = 100
    x = torch.randn(batch_size, seq_len, 3)

    preds = model(x)
    assert preds["power"].shape == (batch_size, seq_len, 4)
    assert preds["on_off"].shape == (batch_size, seq_len, 4)
    assert "transition" in preds
    assert preds["transition"].shape == (batch_size, seq_len, 4)

    # Test MultiApplianceLoss with transition loss
    criterion = MultiApplianceLoss(
        appliances=appliances,
        lambda_bce=1.0,
        lambda_transition=0.5,
    )
    y_power = torch.randn(batch_size, seq_len, 4)
    y_onoff = torch.randint(0, 2, (batch_size, seq_len, 4)).float()
    y_trans = torch.randint(0, 2, (batch_size, seq_len, 4)).float()
    mask = torch.ones((batch_size, 4))

    loss, breakdown = criterion(
        power_pred=preds["power"],
        power_true=y_power,
        onoff_pred=preds["on_off"],
        onoff_true=y_onoff,
        appliance_mask=mask,
        transition_pred=preds["transition"],
        transition_true=y_trans,
    )
    assert loss.item() > 0
    assert "fridge_trans" in breakdown
    assert "microwave_trans" in breakdown

    loss.backward()
    # Check that transition classifier weights received gradients
    for app in appliances:
        assert model.heads[app].transition_classifier.weight.grad is not None


