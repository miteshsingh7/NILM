"""Tests for gradient clipping, NaN-weight guard, and checkpoint safety."""

import pytest
import torch
import torch.nn as nn
from src.utils import save_checkpoint


def test_clip_grad_norm_caps_gradients():
    linear = nn.Linear(10, 1)
    x = torch.randn(5, 10)
    out = linear(x).sum() * 1000.0  # large gradient
    out.backward()

    pre_norm = torch.nn.utils.clip_grad_norm_(linear.parameters(), max_norm=2.0)
    assert pre_norm.item() > 2.0

    post_norm = torch.norm(torch.stack([p.grad.norm() for p in linear.parameters()]))
    assert post_norm.item() <= 2.0001


def test_save_checkpoint_nan_guard(tmp_path):
    linear = nn.Linear(10, 1)
    with torch.no_grad():
        linear.weight[0, 0] = float("nan")

    ckpt_path = tmp_path / "corrupted_model.pt"
    with pytest.raises(RuntimeError, match="FATAL: Attempted to save corrupted checkpoint"):
        save_checkpoint(ckpt_path, model=linear)
