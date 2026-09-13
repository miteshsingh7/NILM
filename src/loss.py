"""Multi-appliance joint loss module with active on-state upweighting.

Addresses the degenerate 'predict-zero' minimum on low-duty-cycle appliances by
upweighting the regression MSE loss during active timesteps (y_onoff == 1).

Loss formulation:
    W_reg_k = 1.0 + (on_weight - 1.0) * y_onoff_k
    MSE_weighted_k = sum(W_reg_k * (p_pred_k - p_true_k)^2) / sum(W_reg_k)
    L_k = MSE_weighted_k + lambda_bce * BCE(o_pred_k, o_true_k)
    Total Loss = sum_k (w_k * L_k)
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """Numerically stable Binary Focal Loss (Lin et al., 2017).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    where p_t = p if y=1 else (1-p)
    and alpha_t = alpha if y=1 else (1-alpha)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * torch.pow((1.0 - p_t), self.gamma)
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class MultiApplianceLoss(nn.Module):
    """Joint multi-task loss with active-state upweighting or Focal Loss, soft gating, and per-appliance weights."""

    def __init__(
        self,
        appliances: List[str],
        lambda_bce: float = 1.0,
        on_weight: Union[float, Dict[str, float]] = 8.0,
        appliance_weights: Optional[Dict[str, float]] = None,
        gated: bool = False,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        focal_appliances: Optional[List[str]] = None,
        lambda_transition: float = 0.0,
        transition_weight: float = 5.0,
    ):
        super().__init__()
        self.appliances = list(appliances)
        self.lambda_bce = lambda_bce
        self.lambda_transition = lambda_transition
        self.transition_weight = transition_weight
        self.use_focal_loss = use_focal_loss
        if isinstance(on_weight, dict):
            self.on_weight = {app: float(on_weight.get(app, 8.0)) for app in self.appliances}
        else:
            self.on_weight = {app: float(on_weight) for app in self.appliances}
        self.appliance_weights = appliance_weights or {app: 1.0 for app in self.appliances}
        self.gated = gated
        # If focal_appliances specified, only those appliances use focal loss (e.g. micr/dish/wash)
        # Appliances not in focal_appliances (e.g. fridge) keep standard BCE
        self.focal_appliances = set(focal_appliances) if focal_appliances is not None else set(self.appliances)
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = float(focal_alpha)
        self.focal_loss_fn = BinaryFocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction="none")

    def forward(
        self,
        power_pred: torch.Tensor,
        power_true: torch.Tensor,
        onoff_pred: torch.Tensor,
        onoff_true: torch.Tensor,
        appliance_mask: Optional[torch.Tensor] = None,
        transition_pred: Optional[torch.Tensor] = None,
        transition_true: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Args:
            power_pred: (batch, length, num_appliances)
            power_true: (batch, length, num_appliances)
            onoff_pred: (batch, length, num_appliances) in [0, 1]
            onoff_true: (batch, length, num_appliances) in {0, 1}
            appliance_mask: Optional (batch, num_appliances) or (num_appliances,) binary mask
                            indicating whether the appliance is valid (1.0) or masked out (0.0).
            transition_pred: Optional (batch, length, num_appliances) transition probability in [0, 1].
            transition_true: Optional (batch, length, num_appliances) binary transition targets in {0, 1}.

        Returns:
            total_loss: Scalar torch tensor for backpropagation.
            breakdown: Dictionary of float losses per appliance for granular logging.
        """
        total_loss = torch.tensor(0.0, device=power_pred.device, dtype=power_pred.dtype)
        breakdown: Dict[str, float] = {}

        # Clamp predictions slightly to avoid log(0) in BCE
        eps = 1e-7
        onoff_pred_clamped = torch.clamp(onoff_pred, eps, 1.0 - eps)

        for i, name in enumerate(self.appliances):
            p_pred = power_pred[..., i]
            p_true = power_true[..., i]
            o_pred = onoff_pred_clamped[..., i]
            o_true = onoff_true[..., i]

            # Resolve per-sample mask for this appliance: shape broadcastable to (batch, length)
            if appliance_mask is not None:
                if appliance_mask.dim() == 1:
                    s_mask = appliance_mask[i].view(1, 1).expand_as(p_pred)
                elif appliance_mask.dim() == 2:
                    s_mask = appliance_mask[:, i].unsqueeze(-1).expand_as(p_pred)
                elif appliance_mask.dim() == 3:
                    s_mask = appliance_mask[..., i]
                else:
                    s_mask = torch.ones_like(p_pred)
            else:
                s_mask = torch.ones_like(p_pred)

            valid_elements = torch.sum(s_mask)
            if valid_elements <= 0:
                # Entire batch is masked out for this appliance
                breakdown[f"{name}_mse"] = 0.0
                breakdown[f"{name}_bce"] = 0.0
                breakdown[f"{name}_loss"] = 0.0
                continue

            # Soft gating: regression output gated by on/off probability
            if self.gated:
                p_pred_reg = p_pred * onoff_pred[..., i]
                p_true_target = p_true * o_true
            else:
                p_pred_reg = p_pred
                p_true_target = p_true

            # Active-state upweighting inside regression loss
            # w_reg is scaled per-sample by s_mask so masked windows contribute strictly 0.0
            app_on_weight = self.on_weight.get(name, 8.0) if isinstance(self.on_weight, dict) else self.on_weight
            w_reg = 1.0 + (app_on_weight - 1.0) * o_true
            effective_w_reg = w_reg * s_mask
            sq_err = (p_pred_reg - p_true_target) ** 2
            denom_reg = torch.sum(effective_w_reg) + 1e-6
            weighted_mse = torch.sum(effective_w_reg * sq_err) / denom_reg

            # Recover logits from the clamped sigmoid output via torch.logit()
            o_logits = torch.logit(o_pred)
            if self.use_focal_loss and name in self.focal_appliances:
                classif_raw = self.focal_loss_fn(o_logits, o_true)
            else:
                classif_raw = F.binary_cross_entropy_with_logits(o_logits, o_true, reduction="none")

            denom_bce = valid_elements + 1e-6
            classif_loss = torch.sum(classif_raw * s_mask) / denom_bce

            app_loss = weighted_mse + self.lambda_bce * classif_loss

            # Auxiliary transition loss
            if self.lambda_transition > 0.0 and transition_pred is not None and transition_true is not None:
                t_pred = transition_pred[..., i]
                t_true = transition_true[..., i]
                t_pred_clamped = torch.clamp(t_pred, eps, 1.0 - eps)
                t_logits = torch.logit(t_pred_clamped)
                w_trans = 1.0 + (self.transition_weight - 1.0) * t_true
                trans_raw = F.binary_cross_entropy_with_logits(t_logits, t_true, weight=w_trans, reduction="none")
                trans_loss = torch.sum(trans_raw * s_mask) / denom_bce
                app_loss = app_loss + self.lambda_transition * trans_loss
                breakdown[f"{name}_trans"] = trans_loss.item()

            weight = self.appliance_weights.get(name, 1.0)
            total_loss = total_loss + weight * app_loss

            breakdown[f"{name}_mse"] = weighted_mse.item()
            breakdown[f"{name}_bce"] = classif_loss.item()
            breakdown[f"{name}_loss"] = app_loss.item()

        breakdown["total_loss"] = total_loss.item()
        return total_loss, breakdown
