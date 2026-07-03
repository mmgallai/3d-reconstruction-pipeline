"""Splatfacto model + depth supervision loss.

Subclasses nerfstudio's SplatfactoModel and:
  - forces `output_depth_during_training=True` so the rasterizer renders depth
  - adds a per-pixel L1 depth loss to get_loss_dict
  - masks out invalid (zero) depth pixels (Femto ToF returns 0 outside range)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Type

import torch

from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig


@dataclass
class SplatfactoTofModelConfig(SplatfactoModelConfig):
    """SplatfactoModelConfig + depth supervision params."""
    _target: Type = field(default_factory=lambda: SplatfactoTofModel)
    output_depth_during_training: bool = True
    """Splatfacto needs to render depth during training for the depth loss."""
    depth_lambda: float = 0.02
    """Weight on the L1 depth loss. 0.0 disables it (= vanilla splatfacto)."""
    depth_min: float = 0.05
    """Pixels with gt depth below this are treated as invalid (in metres)."""
    depth_max: float = 10.0
    """Pixels with gt depth above this are treated as invalid (in metres).
    Outliers (e.g. far-field DA3 estimates) get masked."""


class SplatfactoTofModel(SplatfactoModel):
    """SplatfactoModel + L1 depth loss."""
    config: SplatfactoTofModelConfig

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        depth_lambda = float(self.config.depth_lambda)
        if depth_lambda <= 0.0 or not self.training:
            return loss_dict

        pred_depth = outputs.get("depth", None)
        if pred_depth is None:
            return loss_dict
        gt_depth = batch.get("depth_image", None)
        if gt_depth is None:
            return loss_dict

        # Both are [H, W, 1] in metres (DepthDataset applies depth_unit_scale_factor)
        gt_depth = gt_depth.to(pred_depth.device).float()
        if gt_depth.shape != pred_depth.shape:
            return loss_dict

        valid = (gt_depth > self.config.depth_min) & (gt_depth < self.config.depth_max)
        n_valid = int(valid.sum().item())
        if n_valid < 100:
            return loss_dict

        # L1 on valid pixels only
        diff = torch.abs(pred_depth - gt_depth)[valid]
        depth_loss = diff.mean()

        loss_dict["depth_loss"] = depth_lambda * depth_loss
        return loss_dict
