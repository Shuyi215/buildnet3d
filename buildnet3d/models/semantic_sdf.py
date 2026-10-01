"""
Semantic-SDF: NeuS + 3D semantic segmentation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Type, cast

import numpy as np
import torch
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.dataparsers.base_dataparser import Semantics
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.renderers import SemanticRenderer
from nerfstudio.models.neus import NeuSModel, NeuSModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils.colormaps import ColormapOptions


@dataclass
class SemanticSDFModelConfig(NeuSModelConfig):
    """Semantic-SDF Model Config."""

    _target: Type = field(default_factory=lambda: SemanticSDFModel)
    fg_loss_relat_mult: float = 1.0
    """Factor that multiplies the foreground loss"""
    use_semantics: bool = True
    """Whether to use semantic segmentation."""
    semantic_loss_mult: float = 1.0
    """Factor that multiplies the semantic loss"""


class SemanticSDFModel(NeuSModel):
    """NeuS extended with semantic segmentation."""

    config: SemanticSDFModelConfig

    def __init__(
        self, config: SemanticSDFModelConfig, metadata: Dict, **kwargs
    ) -> None:
        super().__init__(config=config, **kwargs)

        semantics = metadata.get("semantics")
        if self.config.use_semantics:
            assert isinstance(semantics, Semantics)
            self.colormap = semantics.colors.clone().detach().to(self.device)
            self.color_mapping = {
                tuple(np.round(np.array(color), 3)): index
                for index, color in enumerate(semantics.colors.tolist())
            }
        else:
            self.colormap = None
            self.color_mapping = {}
        self._logger = logging.getLogger(__name__)

    def populate_modules(self) -> None:
        """Instantiate modules and fields."""
        super().populate_modules()

        self.field = self.config.sdf_field.setup(
            aabb=self.scene_box.aabb,
            num_images=self.num_train_data,
            use_average_appearance_embedding=(
                self.config.use_average_appearance_embedding
            ),
            spatial_distortion=self.scene_contraction,
        )

        self.renderer_semantics = SemanticRenderer()
        self.semantic_loss = torch.nn.CrossEntropyLoss(reduction="mean")

    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Render RGB / geometry / semantics."""
        samples_and_field_outputs = self.sample_and_forward_field(ray_bundle=ray_bundle)

        field_outputs: Dict[FieldHeadNames, torch.Tensor] = cast(
            Dict[FieldHeadNames, torch.Tensor], samples_and_field_outputs["field_outputs"]
        )
        ray_samples = samples_and_field_outputs["ray_samples"]
        weights = samples_and_field_outputs["weights"]

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        assert (
            ray_bundle.metadata is not None and "directions_norm" in ray_bundle.metadata
        ), "directions_norm is required in ray_bundle.metadata"
        depth = depth / ray_bundle.metadata["directions_norm"]
        normal = self.renderer_normal(
            semantics=field_outputs[FieldHeadNames.NORMALS], weights=weights
        )
        accumulation = self.renderer_accumulation(weights=weights)

        if self.config.background_model != "none":
            self._logger.warning(
                "Background model active; semantic-sdf assumes background_model='none'."
            )

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "normal": normal,
            "weights": weights,
            "directions_norm": ray_bundle.metadata["directions_norm"],
            "field_outputs": field_outputs,
        }

        if FieldHeadNames.SEMANTICS in field_outputs:
            semantics = self.renderer_semantics(
                semantics=field_outputs[FieldHeadNames.SEMANTICS], weights=weights
            )
            outputs["semantics"] = semantics

        if self.training:
            grad_points = field_outputs[FieldHeadNames.GRADIENT]
            outputs.update({"eik_grad": grad_points})
            outputs.update(samples_and_field_outputs)

        outputs["normal_vis"] = (outputs["normal"] + 1.0) / 2.0
        return outputs

    def get_loss_dict(
        self,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        metrics_dict: Optional[Dict[str, Any]] = None,
        step: int = 0,
    ) -> Dict[str, Any]:
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        if "fg_mask_loss" in loss_dict:
            loss_dict["fg_mask_loss"] = (
                self.config.fg_loss_relat_mult * loss_dict["fg_mask_loss"]
            )

        if (
            self.config.use_semantics
            and "semantics" in outputs
            and "semantics" in batch
        ):
            gt_semantics = batch["semantics"][..., 0].long().to(self.device)
            loss_dict["semantics_loss"] = self.config.semantic_loss_mult * self.semantic_loss(
                outputs["semantics"], gt_semantics
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

        if (
            self.config.use_semantics
            and "semantics" in outputs
            and "semantics" in batch
        ):
            gt_semantics = batch["semantics"][..., 0].long().to(self.device)
            pred_semantics = torch.argmax(
                torch.nn.functional.softmax(outputs["semantics"], dim=-1), dim=-1
            )
            images_dict["semantics_colormap"] = self.colormap.to(self.device)[
                pred_semantics
            ]
            semantic_error = (
                (pred_semantics != gt_semantics).unsqueeze(-1).type(torch.float32)
            )
            images_dict["semantics_error"] = colormaps.apply_colormap(
                semantic_error, ColormapOptions(colormap="turbo")
            )

        return metrics_dict, images_dict
