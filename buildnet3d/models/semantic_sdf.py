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
from PIL import Image, ImageDraw, ImageFont
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.dataparsers.base_dataparser import Semantics
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.renderers import SemanticRenderer
from nerfstudio.models.neus import NeuSModel, NeuSModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils.colormaps import ColormapOptions
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class SemanticSDFModelConfig(NeuSModelConfig):
    """Semantic-SDF Model Config
    Extended with RGB uncertainty and semantic uncertainty support. 
    """

    _target: Type = field(default_factory=lambda: SemanticSDFModel)
    fg_loss_relat_mult: float = 1.0
    """Factor that multiplies the foreground loss"""
    
    use_semantics: bool = True
    """Whether to use semantic segmentation."""
    semantic_loss_mult: float = 1.0
    """Factor that multiplies the semantic loss"""
    
    use_rgb_uncertainty: bool = True
    """Enable heteroscedastic RGB uncertainty modeling."""
    use_rgb_uncertainty_warmup: bool = True
    
    rgb_uncertainty_delay_steps: int = 5000
    
    rgb_uncertainty_loss_mult_warmup: float = 0.5

    rgb_uncertainty_loss_mult: float = 1.0
    """Global multiplier for the uncertainty-weighted RGB loss."""
    
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

        # self.step = 0

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
        
        self.step = 0

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
        if self.config.use_rgb_uncertainty and "rgb_uncertainty_logvar" in field_outputs:
            # field_outputs["rgb_uncertainty_logvar"] shape: [R, S, 3]
            per_sample_logvar = field_outputs["rgb_uncertainty_logvar"]
            # Match weighting shape to [R, S, 1] without introducing extra dims
            w = weights
            if w.dim() == per_sample_logvar.dim() - 1:
                w = w.unsqueeze(-1)
            # Weighted sum along sample dimension (-2) similar to RGB renderer
            rgb_logvar = torch.sum(per_sample_logvar * w, dim=-2)
            outputs["rgb_uncertainty_logvar"] = rgb_logvar
            # Also expose variance for downstream visualization/analysis
            outputs["rgb_uncertainty_var"] = torch.exp(rgb_logvar)

        # Aggregate Semantic uncertainty (weighted sum of per-sample log-variance)
        if self.config.use_semantic_uncertainty and "semantic_uncertainty_logvar" in field_outputs:
            # field_outputs["semantic_uncertainty_logvar"] shape: [R, S, C]
            per_sample_logvar = field_outputs["semantic_uncertainty_logvar"]
            # Match weighting shape to [R, S, 1] without introducing extra dims
            w = weights
            if w.dim() == per_sample_logvar.dim() - 1:
                w = w.unsqueeze(-1)
            # Weighted sum along sample dimension (-2) similar to RGB renderer
            semantic_logvar = torch.sum(per_sample_logvar * w, dim=-2)
            outputs["semantic_uncertainty_logvar"] = semantic_logvar
            # Also expose variance for downstream visualization/analysis
            outputs["semantic_uncertainty_var"] = torch.exp(semantic_logvar)
            
            
        if self.training:
            grad_points = field_outputs[FieldHeadNames.GRADIENT]
            outputs.update({"eik_grad": grad_points})
            outputs.update(samples_and_field_outputs)
            if self.config.use_rgb_uncertainty and "rgb_uncertainty_logvar" in field_outputs:
                outputs["rgb_uncertainty_samples"] = field_outputs["rgb_uncertainty_logvar"]
                outputs["rgb_uncertainty_samples_var"] = torch.exp(field_outputs["rgb_uncertainty_logvar"])  # [R,S,3]
            if self.config.use_semantic_uncertainty and "semantic_uncertainty_logvar" in field_outputs:
                outputs["semantic_uncertainty_samples"] = field_outputs["semantic_uncertainty_logvar"]
                outputs["semantic_uncertainty_samples_var"] = torch.exp(field_outputs["semantic_uncertainty_logvar"])  # [R,S,C]

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
        
        if self.training:
              self.step += 1
            
            
        # Start with base losses (includes eikonal etc.)
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)


        # increase weight of fg_mask loss
        if "fg_mask_loss" in loss_dict:
            loss_dict["fg_mask_loss"] = self.config.fg_loss_relat_mult * loss_dict["fg_mask_loss"]
        
        # RGB loss with uncertainty
        if self.config.use_rgb_uncertainty and "rgb_uncertainty_logvar" in outputs:
            image = batch["image"].to(self.device)
            # Blend background same way as base (reuse renderer method)
            pred_image, gt_image = self.renderer_rgb.blend_background_for_loss_computation(
                pred_image=outputs["rgb"],
                pred_accumulation=outputs["accumulation"],
                gt_image=image,
            )
            rgb_logvar = outputs["rgb_uncertainty_logvar"]  # [R,3]
            # Heteroscedastic Gaussian negative log-likelihood (up to constant)
            err2 = (gt_image - pred_image) ** 2
            loss_map = err2 / torch.exp(rgb_logvar) + rgb_logvar
            # rgb_uncertainty_loss = (loss_map * mask).sum() / (mask.sum() + 1e-6) * self.config.rgb_uncertainty_loss_mult + 7.0
            
            
            if self.config.use_rgb_uncertainty_warmup:
                if self.step <= self.config.rgb_uncertainty_delay_steps:
                    rgb_uncertainty_loss = (loss_map.mean() + 7) * self.config.rgb_uncertainty_loss_mult_warmup
                else:
                    rgb_uncertainty_loss = (loss_map.mean() + 7) * self.config.rgb_uncertainty_loss_mult
            else:
                rgb_uncertainty_loss = (loss_map.mean() + 7) * self.config.rgb_uncertainty_loss_mult
            loss_dict["rgb_loss_uncertainty"] = rgb_uncertainty_loss
            loss_dict["rgb_loss"] = self.rgb_loss(pred_image,gt_image)


        # prepare for semantic loss(gt)
        if "semantics" in batch:
            gt_semantics = batch["semantics"][..., 0].long().to(self.device)
        #     bg_semantic_id = 0
        #     valid_mask = gt_semantics != bg_semantic_id  # ignore background pixels
        #     masked_gt_sem = gt_semantics[valid_mask]
        
        # Semantic loss with uncertainty
        if self.config.use_semantic_uncertainty and "semantic_uncertainty_logvar" in outputs:
            if self.step <= self.config.semantic_uncertainty_delay_steps:
                if "semantics" in outputs and "semantics" in batch:
                    loss_dict["semantics_loss"] = self.config.semantic_loss_mult * self.semantic_loss_1(
                        outputs["semantics"], gt_semantics
                    )
            
            else:  
            # Reparameterization trick
                sum_prob = torch.zeros_like(outputs["semantics"])
                for i in range(self.config.N_reparam_samples):
                    noise = torch.randn_like(outputs["semantics"])
                    sample_logit = outputs["semantics"] + noise * torch.exp(0.5 * outputs["semantic_uncertainty_logvar"])
                    sample_prob = torch.softmax(sample_logit, dim=-1)
                    sum_prob += sample_prob
                avg_prob = sum_prob / self.config.N_reparam_samples
                avg_log_prob = torch.log(avg_prob + 1e-8)  # avoid log(0)
                loss_dict["semantics_loss"] = self.config.semantic_loss_mult * self.semantic_loss_2(
                    avg_log_prob, gt_semantics
                )
                
        else:
            # Semantic loss
            if self.config.use_semantics and "semantics" in outputs and "semantics" in batch:
                loss_dict["semantics_loss"] = self.config.semantic_loss_mult * self.semantic_loss_1(
                    outputs["semantics"], gt_semantics
                )
            else:
                pass
        
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, Any], batch: Dict[str, Any]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Add visualization if available."""
        metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)

        image = batch["image"].to(self.device)
        pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )
        
        # RGB error: GT - Pred
        rgb_error = torch.abs(gt_rgb - pred_rgb).mean(dim=-1, keepdim=True)
        images_dict["rgb_error"] = colormaps.apply_colormap(rgb_error, ColormapOptions(colormap="turbo"))
        
        # semantics error: GT - Pred
        if self.config.use_semantics and "semantics" in outputs and "semantics" in batch:
            gt_semantics = batch["semantics"][..., 0].long().to(self.device)
            pred_semantics = torch.argmax(
                torch.nn.functional.softmax(outputs["semantics"], dim=-1), dim=-1
            )
            images_dict["semantics_colormap"] = self.colormap.to(self.device)[
                pred_semantics
            ]
        semantic_error = (pred_semantics != gt_semantics).unsqueeze(-1).type(torch.float32)
        images_dict["semantics_error"] = colormaps.apply_colormap(semantic_error, ColormapOptions(colormap="turbo"))
        
        

        def draw_text_on_image(tensor_img, v_min, v_max):
            """
            tensor_img: (H, W_cbar, 3) Torch Tensor, range [0, 1]
            """
            H, W_cbar, _ = tensor_img.shape
            text_area_width = 60
            text_bg = np.ones((H, text_area_width, 3), dtype=np.uint8) * 255
            cbar_np = (tensor_img.cpu().numpy() * 255).astype(np.uint8)
            full_img_np = np.concatenate([cbar_np, text_bg], axis=1)
            img_pil = Image.fromarray(full_img_np)
            draw = ImageDraw.Draw(img_pil)
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()
            labels = [
                (5, f"{v_max:.3f}"),                   # max
                (H // 2 - 8, f"{(v_max+v_min)/2:.3f}"), # mid
                (H - 20, f"{v_min:.3f}")               # min
            ]
            for y, text in labels:
                draw.text((W_cbar + 5, y), text, fill="black", font=font)
            return torch.from_numpy(np.array(img_pil).astype(np.float32) / 255.0)
        
        def get_colorbar_image(v_min, v_max, height): 
            """Create Colorbar"""
            vals = torch.linspace(1.0, 0.0, height, device=self.device).view(-1, 1)
            options = ColormapOptions(colormap="turbo")
            raw_color = colormaps.apply_colormap(vals, options).view(height, 1, 3)
            width = 20
            cbar = raw_color.repeat(1, width, 1)
            cbar_with_text = draw_text_on_image(cbar, v_min.item(), v_max.item())
            
            return cbar_with_text.to(self.device)
        
        # RGB uncertainty
        if self.config.use_rgb_uncertainty and ("rgb_uncertainty_logvar" in outputs):
            rgb_logvar_map = outputs["rgb_uncertainty_logvar"]
            mean_rgb_logvar = rgb_logvar_map.mean(dim=-1, keepdim=True)
            v_min_rgb = mean_rgb_logvar.min()
            v_max_rgb = mean_rgb_logvar.max()
            norm_logvar_rgb = (mean_rgb_logvar - v_min_rgb) / (v_max_rgb - v_min_rgb + 1e-8)
            H, W, _ = outputs["rgb"].shape
            options_rgb = ColormapOptions(colormap="turbo")
            main_img_rgb = colormaps.apply_colormap(norm_logvar_rgb, options_rgb).view(H, W, 3)
            cbar_rgb = get_colorbar_image(v_min_rgb, v_max_rgb, H)
            sep = torch.ones((H, 4, 3), device=self.device) * 0.2
            images_dict["rgb_uncertainty_var"] = torch.cat([main_img_rgb, sep, cbar_rgb], dim=1)
            
        # semantic uncertainty
        if self.config.use_semantic_uncertainty and ("semantic_uncertainty_logvar" in outputs):
            sem_logvar_map = outputs["semantic_uncertainty_logvar"]
            mean_sem_logvar = sem_logvar_map.mean(dim=-1, keepdim=True)
            v_min_sem = mean_sem_logvar.min()
            v_max_sem = mean_sem_logvar.max()
            norm_logvar_sem = (mean_sem_logvar - v_min_sem) / (v_max_sem - v_min_sem + 1e-8)
            H, W, _ = outputs["rgb"].shape
            options_sem = ColormapOptions(colormap="turbo")
            main_img_sem = colormaps.apply_colormap(norm_logvar_sem, options_sem).view(H, W, 3)
            cbar_sem = get_colorbar_image(v_min_sem, v_max_sem, H)
            sep = torch.ones((H, 4, 3), device=self.device) * 0.2
            images_dict["semantic_uncertainty_var"] = torch.cat([main_img_sem, sep, cbar_sem], dim=1)


        return metrics_dict, images_dict
