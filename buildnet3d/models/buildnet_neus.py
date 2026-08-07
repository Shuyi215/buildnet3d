"""NeuS with BuildNet eval visualizations (e.g. rgb_error)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Type

import torch
from nerfstudio.models.neus import NeuSModel, NeuSModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils.colormaps import ColormapOptions


@dataclass
class BuildNetNeuSModelConfig(NeuSModelConfig):
    """NeuS config with BuildNet image panels."""

    _target: Type = field(default_factory=lambda: BuildNetNeuSModel)
    fg_loss_relat_mult: float = 1.0
    """Factor that multiplies the foreground mask loss."""


class BuildNetNeuSModel(NeuSModel):
    """NeuS plus BuildNet logging / visualization helpers."""

    config: BuildNetNeuSModelConfig

    def get_loss_dict(
        self,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        metrics_dict=None,
    ) -> Dict[str, Any]:
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        if "fg_mask_loss" in loss_dict:
            loss_dict["fg_mask_loss"] = (
                self.config.fg_loss_relat_mult * loss_dict["fg_mask_loss"]
            )
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, Any], batch: Dict[str, Any]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)

        image = batch["image"].to(self.device)
        pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )
        rgb_error = torch.abs(gt_rgb - pred_rgb).mean(dim=-1, keepdim=True)
        images_dict["rgb_error"] = colormaps.apply_colormap(
            rgb_error, ColormapOptions(colormap="turbo")
        )
        return metrics_dict, images_dict
