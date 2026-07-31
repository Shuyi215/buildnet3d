"""
Uncertainty Semantic-SDF: Semantic-SDF + heteroscedastic RGB / semantic uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Type

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.utils import colormaps
from nerfstudio.utils.colormaps import ColormapOptions

from buildnet3d.models.semantic_sdf import SemanticSDFModel, SemanticSDFModelConfig


@dataclass
class UncertaintySemanticSDFModelConfig(SemanticSDFModelConfig):
    """Semantic-SDF with RGB / semantic uncertainty."""

    _target: Type = field(default_factory=lambda: UncertaintySemanticSDFModel)

    use_rgb_uncertainty: bool = True
    """Enable heteroscedastic RGB uncertainty modeling."""
    rgb_uncertainty_delay_steps: int = 5000
    """Number of steps to delay the RGB uncertainty loss."""
    rgb_uncertainty_loss_mult: float = 1.0
    """Global multiplier for the uncertainty-weighted RGB loss."""

    use_semantic_uncertainty: bool = True
    """Enable heteroscedastic semantic uncertainty modeling."""
    semantic_uncertainty_delay_steps: int = 10000
    """Number of steps to delay the semantic uncertainty loss."""
    semantic_uncertainty_loss_mult: float = 1.0
    """Global multiplier for the uncertainty-weighted semantic loss."""
    reparameterize_semantic_uncertainty: bool = True
    """Whether to use the reparameterization trick for semantic uncertainty."""
    N_reparam_samples: int = 10
    """Number of Monte Carlo samples for semantic uncertainty reparameterization."""


class UncertaintySemanticSDFModel(SemanticSDFModel):
    """SemanticSDFModel extended with RGB and semantic uncertainty."""

    config: UncertaintySemanticSDFModelConfig

    def populate_modules(self) -> None:
        super().populate_modules()
        self.semantic_loss_nll = torch.nn.NLLLoss(reduction="mean")
        self.step = 0

    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        outputs = super().get_outputs(ray_bundle)
        field_outputs = outputs["field_outputs"]
        weights = outputs["weights"]

        if (
            self.config.use_rgb_uncertainty
            and "rgb_uncertainty_logvar" in field_outputs
        ):
            per_sample_logvar = field_outputs["rgb_uncertainty_logvar"]
            w = weights
            if w.dim() == per_sample_logvar.dim() - 1:
                w = w.unsqueeze(-1)
            rgb_logvar = torch.sum(per_sample_logvar * w, dim=-2)
            outputs["rgb_uncertainty_logvar"] = rgb_logvar
            outputs["rgb_uncertainty_var"] = torch.exp(rgb_logvar)

        if (
            self.config.use_semantic_uncertainty
            and "semantic_uncertainty_logvar" in field_outputs
        ):
            per_sample_logvar = field_outputs["semantic_uncertainty_logvar"]
            w = weights
            if w.dim() == per_sample_logvar.dim() - 1:
                w = w.unsqueeze(-1)
            semantic_logvar = torch.sum(per_sample_logvar * w, dim=-2)
            outputs["semantic_uncertainty_logvar"] = semantic_logvar
            outputs["semantic_uncertainty_var"] = torch.exp(semantic_logvar)

        if self.training:
            if (
                self.config.use_rgb_uncertainty
                and "rgb_uncertainty_logvar" in field_outputs
            ):
                outputs["rgb_uncertainty_samples"] = field_outputs[
                    "rgb_uncertainty_logvar"
                ]
                outputs["rgb_uncertainty_samples_var"] = torch.exp(
                    field_outputs["rgb_uncertainty_logvar"]
                )
            if (
                self.config.use_semantic_uncertainty
                and "semantic_uncertainty_logvar" in field_outputs
            ):
                outputs["semantic_uncertainty_samples"] = field_outputs[
                    "semantic_uncertainty_logvar"
                ]
                outputs["semantic_uncertainty_samples_var"] = torch.exp(
                    field_outputs["semantic_uncertainty_logvar"]
                )

        return outputs

    def get_loss_dict(
        self,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        metrics_dict: Optional[Dict[str, Any]] = None,
        step: int = 0,
    ) -> Dict[str, Any]:
        if self.training:
            self.step += 1

        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict, step)

        if (
            self.config.use_rgb_uncertainty
            and "rgb_uncertainty_logvar" in outputs
            and self.step > self.config.rgb_uncertainty_delay_steps
        ):
            image = batch["image"].to(self.device)
            pred_image, gt_image = self.renderer_rgb.blend_background_for_loss_computation(
                pred_image=outputs["rgb"],
                pred_accumulation=outputs["accumulation"],
                gt_image=image,
            )
            rgb_logvar = outputs["rgb_uncertainty_logvar"]
            err2 = (gt_image - pred_image) ** 2
            loss_map = err2 / torch.exp(rgb_logvar) + rgb_logvar
            loss_dict["rgb_loss_uncertainty"] = (
                loss_map.mean() + 7
            ) * self.config.rgb_uncertainty_loss_mult
            loss_dict["rgb_loss"] = self.rgb_loss(pred_image, gt_image)

        if (
            self.config.use_semantic_uncertainty
            and "semantic_uncertainty_logvar" in outputs
            and "semantics" in outputs
            and "semantics" in batch
        ):
            gt_semantics = batch["semantics"][..., 0].long().to(self.device)
            if self.step <= self.config.semantic_uncertainty_delay_steps:
                loss_dict["semantics_loss"] = (
                    self.config.semantic_loss_mult
                    * self.semantic_loss(outputs["semantics"], gt_semantics)
                )
            else:
                sum_prob = torch.zeros_like(outputs["semantics"])
                for _ in range(self.config.N_reparam_samples):
                    noise = torch.randn_like(outputs["semantics"])
                    sample_logit = outputs["semantics"] + noise * torch.exp(
                        0.5 * outputs["semantic_uncertainty_logvar"]
                    )
                    sum_prob = sum_prob + torch.softmax(sample_logit, dim=-1)
                avg_prob = sum_prob / self.config.N_reparam_samples
                avg_log_prob = torch.log(avg_prob + 1e-8)
                loss_dict["semantics_loss"] = (
                    self.config.semantic_loss_mult
                    * self.config.semantic_uncertainty_loss_mult
                    * self.semantic_loss_nll(avg_log_prob, gt_semantics)
                )

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, Any], batch: Dict[str, Any]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)

        def draw_text_on_image(tensor_img, v_min, v_max):
            H, W_cbar, _ = tensor_img.shape
            text_area_width = 60
            text_bg = np.ones((H, text_area_width, 3), dtype=np.uint8) * 255
            cbar_np = (tensor_img.cpu().numpy() * 255).astype(np.uint8)
            full_img_np = np.concatenate([cbar_np, text_bg], axis=1)
            img_pil = Image.fromarray(full_img_np)
            draw = ImageDraw.Draw(img_pil)
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except OSError:
                font = ImageFont.load_default()
            labels = [
                (5, f"{v_max:.3f}"),
                (H // 2 - 8, f"{(v_max + v_min) / 2:.3f}"),
                (H - 20, f"{v_min:.3f}"),
            ]
            for y, text in labels:
                draw.text((W_cbar + 5, y), text, fill="black", font=font)
            return torch.from_numpy(np.array(img_pil).astype(np.float32) / 255.0)

        def get_colorbar_image(v_min, v_max, height):
            vals = torch.linspace(1.0, 0.0, height, device=self.device).view(-1, 1)
            options = ColormapOptions(colormap="turbo")
            raw_color = colormaps.apply_colormap(vals, options).view(height, 1, 3)
            cbar = raw_color.repeat(1, 20, 1)
            return draw_text_on_image(cbar, v_min.item(), v_max.item()).to(self.device)

        if self.config.use_rgb_uncertainty and "rgb_uncertainty_logvar" in outputs:
            rgb_logvar_map = outputs["rgb_uncertainty_logvar"]
            mean_rgb_logvar = rgb_logvar_map.mean(dim=-1, keepdim=True)
            v_min_rgb = mean_rgb_logvar.min()
            v_max_rgb = mean_rgb_logvar.max()
            norm_logvar_rgb = (mean_rgb_logvar - v_min_rgb) / (
                v_max_rgb - v_min_rgb + 1e-8
            )
            H, W, _ = outputs["rgb"].shape
            main_img_rgb = colormaps.apply_colormap(
                norm_logvar_rgb, ColormapOptions(colormap="turbo")
            ).view(H, W, 3)
            cbar_rgb = get_colorbar_image(v_min_rgb, v_max_rgb, H)
            sep = torch.ones((H, 4, 3), device=self.device) * 0.2
            images_dict["rgb_uncertainty_var"] = torch.cat(
                [main_img_rgb, sep, cbar_rgb], dim=1
            )

        if (
            self.config.use_semantic_uncertainty
            and "semantic_uncertainty_logvar" in outputs
        ):
            sem_logvar_map = outputs["semantic_uncertainty_logvar"]
            mean_sem_logvar = sem_logvar_map.mean(dim=-1, keepdim=True)
            v_min_sem = mean_sem_logvar.min()
            v_max_sem = mean_sem_logvar.max()
            norm_logvar_sem = (mean_sem_logvar - v_min_sem) / (
                v_max_sem - v_min_sem + 1e-8
            )
            H, W, _ = outputs["rgb"].shape
            main_img_sem = colormaps.apply_colormap(
                norm_logvar_sem, ColormapOptions(colormap="turbo")
            ).view(H, W, 3)
            cbar_sem = get_colorbar_image(v_min_sem, v_max_sem, H)
            sep = torch.ones((H, 4, 3), device=self.device) * 0.2
            images_dict["semantic_uncertainty_var"] = torch.cat(
                [main_img_sem, sep, cbar_sem], dim=1
            )

        return metrics_dict, images_dict
