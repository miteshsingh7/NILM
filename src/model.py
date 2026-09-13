"""Neural network architecture for multi-target seq2seq NILM.

Features:
- Single shared Conv1D + BiLSTM encoder.
- Independent per-appliance dual-branch heads (continuous power + on/off state).
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn


def _make_norm_layer(norm_type: str, channels: int, num_groups: int = 8) -> nn.Module:
    """Factory helper for normalization layers."""
    if norm_type == "batchnorm":
        return nn.BatchNorm1d(channels)
    elif norm_type == "groupnorm":
        groups = min(num_groups, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    elif norm_type == "layernorm":
        return nn.GroupNorm(1, channels)
    else:
        raise ValueError(f"Unknown norm_type: '{norm_type}'. Supported: 'batchnorm', 'groupnorm', 'layernorm'.")


class SharedEncoder(nn.Module):
    """Shared temporal-spectral feature extractor for aggregate mains power."""

    def __init__(
        self,
        in_channels: int = 1,
        conv_filters: Optional[List[int]] = None,
        conv_kernels: Optional[List[int]] = None,
        dropout: float = 0.2,
        lstm_hidden: int = 128,
        norm_type: str = "batchnorm",
        num_groups: int = 8,
    ):
        super().__init__()
        if conv_filters is None:
            conv_filters = [32, 64, 128]
        if conv_kernels is None:
            conv_kernels = [9, 7, 5]

        self.norm_type = norm_type

        # Conv1D layers with padding='same'
        # kernel 9 -> padding 4, kernel 7 -> padding 3, kernel 5 -> padding 2
        self.conv1 = nn.Conv1d(
            in_channels,
            conv_filters[0],
            kernel_size=conv_kernels[0],
            padding=conv_kernels[0] // 2,
        )
        self.bn1 = _make_norm_layer(norm_type, conv_filters[0], num_groups=num_groups)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(
            conv_filters[0],
            conv_filters[1],
            kernel_size=conv_kernels[1],
            padding=conv_kernels[1] // 2,
        )
        self.bn2 = _make_norm_layer(norm_type, conv_filters[1], num_groups=num_groups)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv1d(
            conv_filters[1],
            conv_filters[2],
            kernel_size=conv_kernels[2],
            padding=conv_kernels[2] // 2,
        )
        self.bn3 = _make_norm_layer(norm_type, conv_filters[2], num_groups=num_groups)
        self.relu3 = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

        # Bidirectional LSTM: returns sequence of size 2 * lstm_hidden = 256
        self.lstm = nn.LSTM(
            input_size=conv_filters[2],
            hidden_size=lstm_hidden,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:

            x: Aggregate mains window tensor of shape (batch, length, 1) or (batch, 1, length).

        Returns:
            Encoder features of shape (batch, length, 256).
        """
        # Ensure shape (batch, channels, length) for Conv1D
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, L)
        elif x.dim() == 3 and x.shape[-1] == 1:
            x = x.permute(0, 2, 1)  # (B, L, 1) -> (B, 1, L)

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.relu3(self.bn3(self.conv3(out)))
        out = self.dropout(out)

        # Permute for LSTM: (B, C, L) -> (B, L, C)
        out = out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(out)  # (B, L, 256)
        return lstm_out


class ApplianceHead(nn.Module):
    """Per-appliance disaggregation head with dual regression and on/off branches."""

    def __init__(
        self,
        in_features: int = 256,
        conv_filters: int = 64,
        dense_dim: int = 32,
    ):
        super().__init__()
        # Conv1D feature extractor
        self.conv = nn.Conv1d(
            in_features,
            conv_filters,
            kernel_size=3,
            padding=1,
        )
        self.relu_conv = nn.ReLU()

        # Shared projection dense layer
        self.dense = nn.Linear(conv_filters, dense_dim)
        self.relu_dense = nn.ReLU()

        # Dual branches
        self.power_regressor = nn.Linear(dense_dim, 1)  # Linear activation for normalized power
        self.onoff_classifier = nn.Linear(dense_dim, 1)  # Logits; sigmoid applied in forward

    def forward(self, encoder_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:

            encoder_features: Tensor of shape (batch, length, 256).

        Returns:
            power_pred: (batch, length, 1) continuous normalized power.
            onoff_pred: (batch, length, 1) on-probability in [0, 1].
        """
        # Permute for Conv1D: (B, L, C) -> (B, C, L)
        h = encoder_features.permute(0, 2, 1)
        h = self.relu_conv(self.conv(h))

        # Permute back: (B, C, L) -> (B, L, C)
        h = h.permute(0, 2, 1)
        h = self.relu_dense(self.dense(h))

        power_pred = self.power_regressor(h)
        onoff_prob = torch.sigmoid(self.onoff_classifier(h))

        return power_pred, onoff_prob


class MultiApplianceNILM(nn.Module):
    """Complete Multi-Target Seq2Seq NILM Network.

    Features:
    - 1 shared Conv1D + BiLSTM encoder.
    - N independent dual-branch appliance heads.
    """

    def __init__(
        self,
        appliances: List[str],
        in_channels: int = 1,
        conv_filters: Optional[List[int]] = None,
        conv_kernels: Optional[List[int]] = None,
        dropout: float = 0.2,
        lstm_hidden: int = 128,
        head_conv_filters: int = 64,
        head_dense_dim: int = 32,
        norm_type: str = "batchnorm",
        num_groups: int = 8,
    ):
        super().__init__()
        self.appliances = list(appliances)
        self.appliance_to_idx = {name: i for i, name in enumerate(self.appliances)}

        self.encoder = SharedEncoder(
            in_channels=in_channels,
            conv_filters=conv_filters,
            conv_kernels=conv_kernels,
            dropout=dropout,
            lstm_hidden=lstm_hidden,
            norm_type=norm_type,
            num_groups=num_groups,
        )

        encoder_out_dim = lstm_hidden * 2  # Bidirectional

        self.heads = nn.ModuleDict({
            app_name: ApplianceHead(
                in_features=encoder_out_dim,
                conv_filters=head_conv_filters,
                dense_dim=head_dense_dim,
            )
            for app_name in self.appliances
        })

    def forward(
        self, x: torch.Tensor
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass through shared encoder and all appliance heads.

        Args:
            x: Aggregate input of shape (batch, 599, 1) or (batch, 1, 599).

        Returns:
            Dict containing:
            - 'power_dict': {appliance_name: (batch, 599, 1)}
            - 'onoff_dict': {appliance_name: (batch, 599, 1)}
            - 'power': (batch, 599, num_appliances) stacked tensor
            - 'on_off': (batch, 599, num_appliances) stacked tensor
        """
        features = self.encoder(x)

        power_preds = []
        onoff_preds = []
        power_dict = {}
        onoff_dict = {}

        for name in self.appliances:
            head = self.heads[name]
            p, o = head(features)
            power_dict[name] = p
            onoff_dict[name] = o
            power_preds.append(p)
            onoff_preds.append(o)

        power_stacked = torch.cat(power_preds, dim=-1)  # (B, L, N)
        onoff_stacked = torch.cat(onoff_preds, dim=-1)  # (B, L, N)

        return {
            "power": power_stacked,
            "on_off": onoff_stacked,
            "power_dict": power_dict,
            "onoff_dict": onoff_dict,
        }


class DecoupledTemporalNILM(nn.Module):
    """Architecture for Phase 4: Shared Conv1D front-end + Appliance-Specific BiLSTMs.

    Resolves shared encoder gradient conflict:
    - Conv1D layers remain shared across all appliances (edges, transients, baseline power).
    - BiLSTM layers are strictly decoupled per appliance so rare loads (washing machine, dishwasher)
      are not overridden by dominant gradient updates from high-duty loads (fridge, microwave).
    """

    def __init__(
        self,
        appliances: List[str],
        in_channels: int = 1,
        conv_filters: Optional[List[int]] = None,
        conv_kernels: Optional[List[int]] = None,
        dropout: float = 0.2,
        lstm_hidden: int = 48,
        head_conv_filters: int = 64,
        head_dense_dim: int = 32,
        norm_type: str = "batchnorm",
        num_groups: int = 8,
    ):
        super().__init__()
        self.appliances = list(appliances)
        self.appliance_to_idx = {name: i for i, name in enumerate(self.appliances)}
        self.in_channels = in_channels
        self.lstm_hidden = lstm_hidden

        if conv_filters is None:
            conv_filters = [32, 64, 128]
        if conv_kernels is None:
            conv_kernels = [9, 7, 5]

        # Shared Conv1D Front-End
        self.conv1 = nn.Conv1d(
            in_channels,
            conv_filters[0],
            kernel_size=conv_kernels[0],
            padding=conv_kernels[0] // 2,
        )
        self.bn1 = _make_norm_layer(norm_type, conv_filters[0], num_groups=num_groups)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(
            conv_filters[0],
            conv_filters[1],
            kernel_size=conv_kernels[1],
            padding=conv_kernels[1] // 2,
        )
        self.bn2 = _make_norm_layer(norm_type, conv_filters[1], num_groups=num_groups)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv1d(
            conv_filters[1],
            conv_filters[2],
            kernel_size=conv_kernels[2],
            padding=conv_kernels[2] // 2,
        )
        self.bn3 = _make_norm_layer(norm_type, conv_filters[2], num_groups=num_groups)
        self.relu3 = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

        # Appliance-Specific BiLSTMs
        self.lstms = nn.ModuleDict({
            name: nn.LSTM(
                input_size=conv_filters[2],
                hidden_size=lstm_hidden,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
            )
            for name in self.appliances
        })

        # Appliance-Specific Dual-Branch Heads
        lstm_out_dim = lstm_hidden * 2
        self.heads = nn.ModuleDict({
            name: ApplianceHead(
                in_features=lstm_out_dim,
                conv_filters=head_conv_filters,
                dense_dim=head_dense_dim,
            )
            for name in self.appliances
        })

    def forward(
        self, x: torch.Tensor
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        # Ensure shape (batch, channels, length) for Conv1D
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, L)
        elif x.dim() == 3:
            if x.shape[-1] == self.in_channels:
                x = x.permute(0, 2, 1)  # (B, L, C) -> (B, C, L)

        # Shared temporal convolutional feature map
        c = self.relu1(self.bn1(self.conv1(x)))
        c = self.relu2(self.bn2(self.conv2(c)))
        c = self.relu3(self.bn3(self.conv3(c)))
        c = self.dropout(c)  # (B, 128, L)

        # Permute for LSTM: (B, C, L) -> (B, L, C)
        shared_seq = c.permute(0, 2, 1)

        power_preds = []
        onoff_preds = []
        power_dict = {}
        onoff_dict = {}

        for name in self.appliances:
            lstm_out, _ = self.lstms[name](shared_seq)  # (B, L, 2 * lstm_hidden)
            p, o = self.heads[name](lstm_out)
            power_dict[name] = p
            onoff_dict[name] = o
            power_preds.append(p)
            onoff_preds.append(o)

        power_stacked = torch.cat(power_preds, dim=-1)  # (B, L, N)
        onoff_stacked = torch.cat(onoff_preds, dim=-1)  # (B, L, N)

        return {
            "power": power_stacked,
            "on_off": onoff_stacked,
            "power_dict": power_dict,
            "onoff_dict": onoff_dict,
        }


def build_shared_model(
    appliances: List[str],
    dropout: float = 0.2,
    lstm_hidden: int = 128,
    norm_type: str = "batchnorm",
    num_groups: int = 8,
) -> MultiApplianceNILM:
    """Factory helper to build a MultiApplianceNILM model."""
    return MultiApplianceNILM(
        appliances=appliances,
        dropout=dropout,
        lstm_hidden=lstm_hidden,
        norm_type=norm_type,
        num_groups=num_groups,
    )


def build_decoupled_temporal_model(
    appliances: List[str],
    in_channels: int = 1,
    dropout: float = 0.2,
    lstm_hidden: int = 48,
    norm_type: str = "batchnorm",
    num_groups: int = 8,
) -> DecoupledTemporalNILM:
    """Factory helper to build a DecoupledTemporalNILM model."""
    return DecoupledTemporalNILM(
        appliances=appliances,
        in_channels=in_channels,
        dropout=dropout,
        lstm_hidden=lstm_hidden,
        norm_type=norm_type,
        num_groups=num_groups,
    )
