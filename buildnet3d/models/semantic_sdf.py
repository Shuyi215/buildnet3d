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
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.dataparsers.base_dataparser import Semantics
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.renderers import SemanticRenderer
from nerfstudio.models.neus import NeuSModel, NeuSModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class SemanticSDFModelConfig(NeuSModelConfig):
    """Semantic-SDF Model Config

    Extended with RGB uncertainty support. The uncertainty is modeled as a
    per-sample log-variance and aggregated along the ray using volume weights.
    Loss is replaced by a heteroscedastic likelihood when enabled.
    """

    _target: Type = field(default_factory=lambda: SemanticSDFModel)
    eikonal_loss_mult: float = 5
    """Factor that multiplies the eikonal loss"""
    # fg_loss_mult: float = 0.1
    # """Factor that multiplies the foreground loss"""
    
    use_semantic: bool = True
    """Whether to use semantic segmentation."""
    semantic_loss_mult: float = 1.0
    """Factor that multiplies the semantic loss"""
    eikonal_loss_mult: float = 0.1
    """Factor that multiplies the eikonal loss"""
    fg_loss_mult: float = 0.01
    """Factor that multiplies the foreground loss"""
    rgb_uncertainty_loss_mult: float = 1.0
    """Global multiplier for the uncertainty-weighted RGB loss."""
    use_rgb_uncertainty: bool = True
    """Enable heteroscedastic RGB uncertainty modeling."""
    use_semantic_uncertainty: bool = True
    """Enable heteroscedastic semantic uncertainty modeling."""
    semantic_uncertainty_delay_steps: int = 10000
    """Number of steps to delay the semantic uncertainty loss."""
    reparameterize_semantic_uncertainty: bool = True
    """Whether to use the reparameterization trick for semantic uncertainty."""
    N_reparam_samples: int = 10
    """Number of samples for Monte Carlo estimation when using reparameterization."""
    


class SemanticSDFModel(NeuSModel):
    """SemanticSDFModel extends NeuSModel and (optionally) adds semantic segmentation
    and RGB uncertainty prediction (heteroscedastic regression)."""

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
        self.rgb_loss = torch.nn.MSELoss(reduction="mean")
        self.semantic_loss_1 = torch.nn.CrossEntropyLoss(reduction="mean")
        self.semantic_loss_2 = torch.nn.NLLLoss(reduction="mean")

    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Custom forward to inject RGB uncertainty aggregation after standard NeuS rendering."""
        # Reuse NeuS sampling logic
        samples_and_field_outputs = self.sample_and_forward_field(ray_bundle=ray_bundle)

        field_outputs: Dict[FieldHeadNames, torch.Tensor] = cast(
            Dict[FieldHeadNames, torch.Tensor], samples_and_field_outputs["field_outputs"]
        )
        ray_samples = samples_and_field_outputs["ray_samples"]
        weights = samples_and_field_outputs["weights"]
        bg_transmittance = samples_and_field_outputs["bg_transmittance"]

        # Standard surface rendering logic (copied from SurfaceModel.get_outputs)
        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        assert (
            ray_bundle.metadata is not None and "directions_norm" in ray_bundle.metadata
        ), "directions_norm is required in ray_bundle.metadata"
        depth = depth / ray_bundle.metadata["directions_norm"]
        normal = self.renderer_normal(semantics=field_outputs[FieldHeadNames.NORMALS], weights=weights)
        accumulation = self.renderer_accumulation(weights=weights)

        # Background blending disabled if background_model == "none" (like base)
        if self.config.background_model != "none":
            # We defer to parent implementation if needed; for now assume none for semantic-sdf use-case.
            self._logger.warning("Background model active; current uncertainty pipeline does not alter background.")

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "normal": normal,
            "weights": weights,
            "directions_norm": ray_bundle.metadata["directions_norm"],
        }
        
        # add semantics output
        if FieldHeadNames.SEMANTICS in field_outputs:
            semantics = self.renderer_semantics(
                semantics=field_outputs[FieldHeadNames.SEMANTICS], weights=weights
            )
            outputs["semantics"] = semantics

        # Aggregate RGB uncertainty (weighted sum of per-sample log-variance)
        if self.config.use_rgb_uncertainty and "rgb_uncertainty" in field_outputs:
            # field_outputs["rgb_uncertainty"] shape: [R, S, 3]
            per_sample_logvar = field_outputs["rgb_uncertainty"]
            # Match weighting shape to [R, S, 1] without introducing extra dims
            w = weights
            if w.dim() == per_sample_logvar.dim() - 1:
                w = w.unsqueeze(-1)
            # Weighted sum along sample dimension (-2) similar to RGB renderer
            rgb_logvar = torch.sum(per_sample_logvar * w, dim=-2)
            outputs["rgb_uncertainty"] = rgb_logvar
            # Also expose variance for downstream visualization/analysis
            outputs["rgb_uncertainty_var"] = torch.exp(rgb_logvar)

        # Aggregate Semantic uncertainty (weighted sum of per-sample log-variance)
        if self.config.use_semantic_uncertainty and "semantic_uncertainty" in field_outputs:
            # field_outputs["semantic_uncertainty"] shape: [R, S, C]
            per_sample_logvar = field_outputs["semantic_uncertainty"]
            # Match weighting shape to [R, S, 1] without introducing extra dims
            w = weights
            if w.dim() == per_sample_logvar.dim() - 1:
                w = w.unsqueeze(-1)
            # Weighted sum along sample dimension (-2) similar to RGB renderer
            semantic_logvar = torch.sum(per_sample_logvar * w, dim=-2)
            outputs["semantic_uncertainty"] = semantic_logvar
            # Also expose variance for downstream visualization/analysis
            outputs["semantic_uncertainty_var"] = torch.exp(semantic_logvar)
            
            
        if self.training:
            grad_points = field_outputs[FieldHeadNames.GRADIENT]
            outputs.update({"eik_grad": grad_points})
            outputs.update(samples_and_field_outputs)
            if self.config.use_rgb_uncertainty and "rgb_uncertainty" in field_outputs:
                outputs["rgb_uncertainty_samples"] = field_outputs["rgb_uncertainty"]
                outputs["rgb_uncertainty_samples_var"] = torch.exp(field_outputs["rgb_uncertainty"])  # [R,S,3]
            if self.config.use_semantic_uncertainty and "semantic_uncertainty" in field_outputs:
                outputs["semantic_uncertainty_samples"] = field_outputs["semantic_uncertainty"]
                outputs["semantic_uncertainty_samples_var"] = torch.exp(field_outputs["semantic_uncertainty"])  # [R,S,C]

        # Normal visualization
        outputs["normal_vis"] = (outputs["normal"] + 1.0) / 2.0
        return outputs

    def get_loss_dict(
        self,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        metrics_dict: Optional[Dict[str, Any]] = None,
        step: int = 0,
    ) -> Dict[str, Any]:
        """Extend loss with heteroscedastic RGB uncertainty if enabled."""
        # Start with base losses (includes eikonal etc.)
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)


        # # increase weight of eikonal loss
        # if "eikonal_loss" in loss_dict:
        #     loss_dict["eikonal_loss"] = self.config.eikonal_loss_mult * loss_dict["eikonal_loss"]
        
        # RGB loss with uncertainty
        if self.config.use_rgb_uncertainty and "rgb_uncertainty" in outputs:
            image = batch["image"].to(self.device)
            # Blend background same way as base (reuse renderer method)
            pred_image, gt_image = self.renderer_rgb.blend_background_for_loss_computation(
                pred_image=outputs["rgb"],
                pred_accumulation=outputs["accumulation"],
                gt_image=image,
            )
            rgb_logvar = outputs["rgb_uncertainty"]  # [R,3]
            # Heteroscedastic Gaussian negative log-likelihood (up to constant)
            err2 = (gt_image - pred_image) ** 2
            rgb_uncertainty_loss = (
                err2 / torch.exp(rgb_logvar) + rgb_logvar
            ).mean() * self.config.rgb_uncertainty_loss_mult
            # Replace standard rgb_loss
            loss_dict["rgb_loss_uncertainty"] = rgb_uncertainty_loss
            # loss_dict["rgb_loss"] = rgb_uncertainty_loss
            loss_dict["rgb_loss"] = self.rgb_loss(pred_image, gt_image)  


        # Semantic loss
        if self.config.use_semantic and "semantics" in outputs and "semantics" in batch:
            loss_dict["semantics_loss"] = self.config.semantic_loss_mult * self.semantic_loss_1(
                outputs["semantics"], batch["semantics"][..., 0].long().to(self.device)
            )
        
        
        # Semantic loss with uncertainty
        if self.config.use_semantic_uncertainty and "semantic_uncertainty" in outputs:
            if step <= self.config.semantic_uncertainty_delay_steps:
                if "semantics" in outputs and "semantics" in batch:
                    loss_dict["semantics_loss"] = self.config.semantic_loss_mult * self.semantic_loss_1(
                        outputs["semantics"], batch["semantics"][..., 0].long().to(self.device)
                    )
            
            else:  
                # Reparameterization trick
                sum_prob = torch.zeros_like(outputs["semantics"])
                for i in range(self.config.N_reparam_samples):
                    noise = torch.randn_like(outputs["semantics"])
                    sample_logit = outputs["semantics"] + noise * torch.exp(0.5 * outputs["semantic_uncertainty"])
                    sample_prob = torch.softmax(sample_logit, dim=-1)
                    sum_prob += sample_prob
                avg_prob = sum_prob / self.config.N_reparam_samples
                avg_log_prob = torch.log(avg_prob + 1e-8)  # avoid log(0)
                loss_dict["semantics_loss"] = self.config.semantic_loss_mult * self.semantic_loss_2(
                    avg_log_prob, batch["semantics"][..., 0].long().to(self.device)
                )
                    
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, Any], batch: Dict[str, Any]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Add visualization for RGB uncertainty (mean log-variance) if available."""
        metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)

        # semantics
        semantic_labels = torch.argmax(
            torch.nn.functional.softmax(outputs["semantics"], dim=-1), dim=-1
        )
        images_dict["semantics_colormap"] = self.colormap.to(self.device)[
            semantic_labels
        ]
        
        # RGB Uncertainty
        if self.config.use_rgb_uncertainty and ("rgb_uncertainty_var" in outputs):
            var_map = outputs.get("rgb_uncertainty_var", torch.exp(outputs["rgb_uncertainty"]))  # [R,3]
            mean_var = var_map.mean(dim=-1, keepdim=True).view(-1,1)
            rgb_pred = outputs["rgb"]  # [H,W,3]
            H, Wp, _ = rgb_pred.shape
            mean_var_colormap = colormaps.apply_colormap(mean_var).view(H, Wp, 3)
            images_dict["rgb_uncertainty_var"] = mean_var_colormap
        
        # Semantic Uncertainty
        if self.config.use_semantic_uncertainty and ("semantic_uncertainty_var" in outputs):
            sem_var_map = outputs.get("semantic_uncertainty_var", torch.exp(outputs["semantic_uncertainty"]))  # [R,C]
            mean_sem_var = sem_var_map.mean(dim=-1, keepdim=True).view(-1,1)
            rgb_pred = outputs["rgb"]  # [H,W,3]
            H, Wp, _ = rgb_pred.shape
            mean_sem_var_colormap = colormaps.apply_colormap(mean_sem_var).view(H, Wp, 3)
            images_dict["semantic_uncertainty_var"] = mean_sem_var_colormap

        return metrics_dict, images_dict
