# semantic_sdf.py


"""
Implementation of Semantic-SDF. This model is built on top of NeusFacto and adds
a 3D semantic segmentation model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Type, cast

import numpy as np
import torch
import torch.nn.functional as F
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.dataparsers.base_dataparser import Semantics
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.renderers import SemanticRenderer
from nerfstudio.models.neus import NeuSModel, NeuSModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.model_components.renderers import UncertaintyRenderer

@dataclass
class SemanticSDFModelConfig(NeuSModelConfig):
    """Semantic-SDF Model Config"""

    _target: Type = field(default_factory=lambda: SemanticSDFModel)

    semantic_loss_mult: float = 1.0
    """Factor that multiplies the semantic loss"""

    density_loss_mult: float = 0.01
    """Strength for regularization to force sparse density."""

    # beta_min: float = 0.01
    # """Minimum value for uncertainty."""

    rendered_uncertainty_eps: float = 1e-6
    """Value for clamping the rendered uncertainty (variance) when computing NLL."""

    rgb_uncertainty_loss_mult: float = 1.0
    """Multiplier for RGB uncertainty NLL loss."""
    
    rgb_beta_min: float = 0.01
    """Minimum value for RGB uncertainty to avoid numerical issues."""

class SemanticSDFModel(NeuSModel):
    """SemanticSDFModel extends NeuSFactoModel to add semantic segmentation in 3D."""

    config: SemanticSDFModelConfig

    def __init__(
        self, config: SemanticSDFModelConfig, metadata: Dict, **kwargs
    ) -> None:
        """
        To setup the model, provide a model `config` and the `metadata` from the
        outputs of the dataparser.
        """
        super().__init__(config=config, **kwargs)

        assert "semantics" in metadata.keys() and isinstance(
            metadata["semantics"], Semantics
        )
        self.colormap = metadata["semantics"].colors.clone().detach().to(self.device)

        self.color_mapping = {
            tuple(np.round(np.array(color), 3)): index
            for index, color in enumerate(metadata["semantics"].colors.tolist())
        }
        self._logger = logging.getLogger(__name__)

        self.step = 0

    def populate_modules(self) -> None:
        """Instantiate modules and fields, including proposal networks."""
        super().populate_modules()

        # Fields
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
        self.renderer_uncertainty = UncertaintyRenderer()


    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Pass the `ray_bundle` through the model's field and renderer to get
        the model's output."""
        outputs = super().get_outputs(ray_bundle)
        
        # get field outputs for points
        samples_and_field_outputs = self.sample_and_forward_field(ray_bundle=ray_bundle)

        field_outputs: Dict[FieldHeadNames, torch.Tensor] = cast(
            Dict[FieldHeadNames, torch.Tensor],
            samples_and_field_outputs["field_outputs"],
        )

        outputs["semantics"] = self.renderer_semantics(
            field_outputs[FieldHeadNames.SEMANTICS], weights=outputs["weights"]
        )

        # semantics colormaps
        semantic_labels = torch.argmax(
            torch.nn.functional.softmax(outputs["semantics"], dim=-1), dim=-1
        )
        outputs["semantics_colormap"] = self.colormap.to(semantic_labels.device)[semantic_labels]


        # RGB uncertainty rendering - single channel
        if FieldHeadNames.RGB_UNCERTAINTY in field_outputs:
            point_rgb_uncertainty = field_outputs[FieldHeadNames.RGB_UNCERTAINTY]

            # numerical stability (refer to ActiveNerfacto)
            if torch.isnan(point_rgb_uncertainty).any():
                point_rgb_uncertainty = torch.nan_to_num(point_rgb_uncertainty, 0.0)

            # ActiveNeRF Eq(9)
            rgb_var = self.renderer_uncertainty(
                betas=point_rgb_uncertainty, 
                weights=outputs["weights"]**2
            )
            
            outputs["rendered_rgb_uncertainty"] = rgb_var
            outputs["rendered_rgb_std"] = torch.sqrt(rgb_var)
        else:
            # if no rgb uncertainty head, set to zeros
            outputs["rendered_rgb_uncertainty"] = torch.zeros_like(outputs["rgb"][..., :1])
            outputs["rendered_rgb_std"] = torch.zeros_like(outputs["rgb"][..., :1])



        # # render uncertainty
        # point_semantic_uncertainty = field_outputs["semantic_uncertainty"]
        # # Fix shape mismatch between betas and weights
        # if point_semantic_uncertainty.shape != outputs["weights"].shape:
        #     point_semantic_uncertainty = point_semantic_uncertainty.reshape(outputs["weights"].shape)
        # rendered_semantic_uncertainty = self.renderer_uncertainty(
        #     betas=point_semantic_uncertainty,
        #     weights=outputs["weights"]**2
        # )
        # outputs["rendered_semantic_uncertainty"] = rendered_semantic_uncertainty
        # outputs["rendered_semantic_std"] = torch.sqrt(rendered_semantic_uncertainty)

        return outputs

    def get_loss_dict(
        self,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        metrics_dict: Optional[Dict[str, Any]] = None,
        step: int = 0,
    ) -> Dict[str, Any]:
        """Compute the loss dictionary from the `outputs` of the model, the `batch`
        that contains the ground truth data and the `metrics_dict`."""
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        # -------------------------------------------------------------
        # Alpha-masked losses (Step 3)
        # -------------------------------------------------------------
        # Expect batch["image"] to be RGBA; fall back gracefully if RGB only.
        gt_image = batch["image"].to(self.device)  # [..., C]
        if gt_image.shape[-1] == 4:
            gt_rgb = gt_image[..., :3]
            gt_alpha = gt_image[..., 3:4]
        else:
            gt_rgb = gt_image
            # If no alpha channel provided, treat all pixels as foreground.
            gt_alpha = torch.ones_like(gt_rgb[..., 0:1])

        # foreground weight (can be soft if alpha isn't binary)
        fg_weight = gt_alpha.squeeze(-1)  # [...]
        weight_sum = fg_weight.sum().clamp_min(1.0)

        # ----------------- Semantic loss (masked) -----------------
        semantic_targets = batch["semantics"][..., 0].long().to(self.device)
        semantic_logits = outputs["semantics"]
        # per-pixel cross entropy
        per_pixel_sem_loss = F.cross_entropy(
            semantic_logits, semantic_targets, reduction="none"
        )
        masked_sem_loss = (per_pixel_sem_loss * fg_weight).sum() / weight_sum
        loss_dict["semantics_loss"] = self.config.semantic_loss_mult * masked_sem_loss

        # ----------------- RGB uncertainty NLL (masked) -----------------
        pred_rgb = outputs["rgb"]
        # Ensure shape alignment (flatten if necessary)
        if pred_rgb.shape[-1] != 3:
            raise ValueError(
                f"Expected predicted RGB with 3 channels, got shape {pred_rgb.shape}"
            )

        # squared error per-pixel (sum over channel) then mask
        rgb_sq_error = ((pred_rgb - gt_rgb) ** 2).sum(dim=-1)  # [...]
        masked_rgb_sq_error = rgb_sq_error * fg_weight

        rgb_uncertainty = torch.maximum(
            outputs["rendered_rgb_uncertainty"].squeeze(-1),
            torch.tensor(self.config.rendered_uncertainty_eps, device=self.device),
        )
        # apply same mask to uncertainty terms
        nll_terms = 0.5 * torch.log(rgb_uncertainty) + 0.5 * masked_rgb_sq_error / rgb_uncertainty
        rgb_uncertainty_nll = nll_terms.sum() / weight_sum
        loss_dict["rgb_uncertainty_nll_loss"] = (
            self.config.rgb_uncertainty_loss_mult * rgb_uncertainty_nll
        )

        # ----------------- Density regularization (masked) -----------------
        if "density" in outputs:
            # outputs["density"] may have trailing channel dim of 1
            density_vals = outputs["density"].squeeze(-1)
            masked_density = (density_vals * fg_weight).sum() / weight_sum
            loss_dict["density_reg_loss"] = (
                self.config.density_loss_mult * masked_density
            )

        # add debug stats for monitoring
        loss_dict["alpha_foreground_fraction"] = fg_weight.mean()
        loss_dict["alpha_weight_sum"] = weight_sum

        # #semantic uncertainty NLL loss
        # gt_semantics = batch["semantics"][..., 0].long().to(self.device)
        # pred_semantics_logits = outputs["semantics"]
        # semantic_probs = torch.nn.functional.softmax(pred_semantics_logits, dim=-1)
        # get_one_hot = torch.nn.functional.one_hot(gt_semantics,num_classes=semantic_probs.shape[-1])
        # semantic_error = torch.mean((semantic_probs - get_one_hot.float())**2, dim=-1)

        # uncert = torch.maximum(outputs["rendered_semantic_uncertainty"].squeeze(-1),
        #                        torch.tensor(self.config.rendered_uncertainty_eps,device=self.device))
        
        # loss_dict["semantic_uncertainty_nll_loss"] = torch.mean(
        #     0.5 * torch.log(uncert) + 0.5 * semantic_error / uncert
        # )

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, Any], batch: Dict[str, Any]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Compute image metrics and images from the `outputs` of the model and
        the `batch` which contains input and ground truth data."""
        metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)

        # semantics
        semantic_labels = torch.argmax(
            torch.nn.functional.softmax(outputs["semantics"], dim=-1), dim=-1
        )
        images_dict["semantics_colormap"] = self.colormap.to(self.device)[
            semantic_labels
        ]

        # RGB uncertainty visualization
        rgb_std = outputs["rendered_rgb_std"]
        rgb_uncertainty_colormap = colormaps.apply_colormap(rgb_std)
        images_dict["rgb_uncertainty_colormap"] = rgb_uncertainty_colormap

        # Add RGB uncertainty to metrics
        metrics_dict["rgb_uncertainty_mean"] = torch.mean(outputs["rendered_rgb_uncertainty"]).item()
        metrics_dict["rgb_uncertainty_std"] = torch.std(outputs["rendered_rgb_uncertainty"]).item()


        # # semantic uncertainty
        # semantic_std = outputs["rendered_semantic_std"]
        # semantic_uncertainty_colormap = colormaps.apply_colormap(semantic_std)
        # images_dict["semantic_uncertainty_colormap"] = semantic_uncertainty_colormap

        return metrics_dict, images_dict
