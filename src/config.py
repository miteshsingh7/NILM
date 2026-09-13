"""Configuration and hyperparameter settings for NILM multi-target seq2seq system."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import torch


# Default target appliances and on/off power thresholds in Watts
DEFAULT_APPLIANCE_THRESHOLDS: Dict[str, float] = {
    "fridge": 50.0,
    "microwave": 200.0,
    "dishwasher": 10.0,
    "washing_machine": 20.0,
    "kettle": 2000.0,  # UK-DALE only
}

# Appliance-specific active-state upweight multipliers in MSE loss,
# scaled inversely to each appliance's positive-class rate (fridge: 21.97%, wash: 1.22%, dish: 0.78%, micr: 0.31%).
# Rare bursty appliances (p <= 1.22%) receive 8.0x upweighting;
# high-duty-cycle fridge is scaled inversely: 8.0 * (0.012176 / 0.219698) = 1.44x.
DEFAULT_ON_WEIGHTS: Dict[str, float] = {
    "fridge": 1.44,
    "microwave": 8.0,
    "dishwasher": 8.0,
    "washing_machine": 8.0,
}

# Standard 4-appliance benchmark set present in both REDD and UK-DALE
DEFAULT_APPLIANCES: List[str] = [
    "fridge",
    "microwave",
    "dishwasher",
    "washing_machine",
]

# REDD dataset canonical channel configurations per house
# Mains is always ch1 + ch2 in REDD low_freq
REDD_CHANNEL_MAP: Dict[int, Dict[str, List[int]]] = {
    1: {
        "mains": [1, 2],
        "fridge": [5],
        "dishwasher": [6],
        "microwave": [11],
        "washing_machine": [10, 19, 20],  # washer dryer
    },
    2: {
        "mains": [1, 2],
        "fridge": [9],
        "dishwasher": [10],
        "microwave": [6],
        "washing_machine": [7],
    },
    3: {
        "mains": [1, 2],
        "fridge": [7],
        "dishwasher": [9],
        "microwave": [16],
        "washing_machine": [13, 14],
    },
    4: {
        "mains": [1, 2],
        "fridge": [],
        "dishwasher": [15],
        "microwave": [],
        "washing_machine": [7],
    },
    5: {
        "mains": [1, 2],
        "fridge": [18],
        "dishwasher": [20],
        "microwave": [3],
        "washing_machine": [8, 9],
    },
    6: {
        "mains": [1, 2],
        "fridge": [8],
        "dishwasher": [9],
        "microwave": [],
        "washing_machine": [4],
    },
}

UKDALE_CHANNEL_MAP: Dict[int, Dict[str, List[int]]] = {
    1: {
        "mains": [1],
        "fridge": [12],
        "dishwasher": [6],
        "microwave": [13],
        "washing_machine": [5],
        "kettle": [10],
    },
    2: {
        "mains": [1],
        "fridge": [14],
        "dishwasher": [13],
        "microwave": [15],
        "washing_machine": [12],
        "kettle": [8],
    },
    3: {
        "mains": [1],
        "fridge": [],
        "dishwasher": [],
        "microwave": [],
        "washing_machine": [],
        "kettle": [2],
    },
    4: {
        "mains": [1],
        "fridge": [],
        "dishwasher": [],
        "microwave": [],
        "washing_machine": [],
        "kettle": [],
    },
    5: {
        "mains": [1],
        "fridge": [19],
        "dishwasher": [22],
        "microwave": [23],
        "washing_machine": [24],
        "kettle": [18],
    },
}


@dataclass
class NILMConfig:
    """Master configuration for NILM training, preprocessing, and inference."""

    # Preprocessing
    sample_period_seconds: int = 6
    window_length: int = 599  # ~1 hour at 6-second sampling
    train_stride: int = 599 // 4  # 149 samples (~75% overlap)
    val_test_stride: int = 599  # Non-overlapping for evaluation
    max_gap_fill_samples: int = 3  # Maximum consecutive NaNs to forward-fill (18s)

    # Appliances
    appliances: List[str] = field(default_factory=lambda: list(DEFAULT_APPLIANCES))
    thresholds: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_APPLIANCE_THRESHOLDS))

    # Cross-household split
    dataset_name: str = "redd"  # "redd" or "ukdale"
    held_out_house: int = 1     # House reserved purely for unseen generalization testing
    train_val_split_ratio: float = 0.8  # 80% train, 20% val for remaining houses

    # Model architecture
    conv_filters: List[int] = field(default_factory=lambda: [32, 64, 128])
    conv_kernels: List[int] = field(default_factory=lambda: [9, 7, 5])
    encoder_dropout: float = 0.2
    lstm_hidden_size: int = 128
    head_conv_filters: int = 64
    head_dense_dim: int = 32
    in_channels: int = 1
    norm_type: str = "batchnorm"  # "batchnorm", "groupnorm", "layernorm"
    model_type: str = "shared"    # "shared" or "decoupled_temporal"
    num_groups: int = 8           # Group count for GroupNorm

    # Training
    batch_size: int = 128
    lr: float = 1e-3
    lambda_loss: float = 1.0  # Weight for BCE classification loss
    on_weight: Union[float, Dict[str, float]] = field(default_factory=lambda: dict(DEFAULT_ON_WEIGHTS))  # Active upweight
    appliance_loss_weights: Optional[Dict[str, float]] = None
    epochs: int = 35
    early_stopping_patience: int = 8
    lr_reduce_patience: int = 4
    lr_reduce_factor: float = 0.5
    min_lr: float = 1e-6
    mixed_precision: bool = False  # Disabled: BCE is incompatible with CUDA AMP
    use_focal_loss: bool = False  # If True, replaces on_weight with Binary Focal Loss
    focal_gamma: float = 2.0      # Focusing parameter gamma
    focal_alpha: float = 0.25     # Balance parameter alpha
    focal_appliances: Optional[List[str]] = None  # Specific appliances for focal loss

    # Device resolution: check capability >= 7.0 to prevent sm_60 crash on PyTorch 2.10+
    @staticmethod
    def _default_device() -> str:
        if torch.cuda.is_available():
            try:
                if torch.cuda.get_device_capability(0) >= (7, 0):
                    return "cuda"
                print("Warning: GPU compute capability < 7.0 (sm_60) detected. PyTorch cu128 requires sm_70+. Defaulting to CPU.")
                return "cpu"
            except Exception:
                return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    device: str = field(default_factory=_default_device)

    # Paths
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"

    def get_threshold(self, appliance: str) -> float:
        """Return the on/off threshold in Watts for a given appliance."""
        return self.thresholds.get(appliance, 20.0)

    @property
    def num_appliances(self) -> int:
        return len(self.appliances)
