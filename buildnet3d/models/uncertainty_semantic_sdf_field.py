"""
Uncertainty-augmented semantic-SDF field: RGB / semantic heteroscedastic heads.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Type

import torch
from jaxtyping import Float
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from torch import Tensor, nn

from buildnet3d.models.semantic_sdf_field import SemanticSDFField, SemanticSDFFieldConfig


@dataclass
class UncertaintySemanticSDFFieldConfig(SemanticSDFFieldConfig):
    """Semantic-SDF field config with RGB / semantic uncertainty heads."""

    _target: Type = field(default_factory=lambda: UncertaintySemanticSDFField)

    use_rgb_uncertainty: bool = True
    """Whether to predict per-sample RGB log-variance."""
    rgb_uncertainty_mlp_layers: int = 2
    """Number of hidden layers for RGB uncertainty MLP."""
    rgb_uncertainty_mlp_width: int = 64
    """Width of hidden layers for RGB uncertainty MLP."""
    rgb_beta_min: float = 0.001
    """Minimum beta value for RGB uncertainty variance."""

    use_semantic_uncertainty: bool = True
    """Whether to predict per-sample semantic log-variance."""
    semantic_uncertainty_mlp_layers: int = 2
    """Number of hidden layers for semantic uncertainty MLP."""
    semantic_uncertainty_mlp_width: int = 64
    """Width of hidden layers for semantic uncertainty MLP."""
    semantic_beta_min: float = 0.001
    """Minimum beta value for semantic uncertainty variance."""


class UncertaintySemanticSDFField(SemanticSDFField):
    """SemanticSDFField with optional RGB and semantic uncertainty heads."""

    config: UncertaintySemanticSDFFieldConfig

    def __init__(
        self,
        config: UncertaintySemanticSDFFieldConfig,
        aabb: Float[Tensor, "2 3"],  # noqa: F722
        num_images: int,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
    ) -> None:
        super().__init__(
            config,
            aabb,
            num_images,
            use_average_appearance_embedding,
            spatial_distortion,
        )

        if self.config.use_rgb_uncertainty:
            self.mlp_rgb_uncertainty = MLP(
                in_dim=self.config.geo_feat_dim,
                layer_width=self.config.rgb_uncertainty_mlp_width,
                num_layers=self.config.rgb_uncertainty_mlp_layers,
                activation=nn.ReLU(),
                out_activation=None,
            )
            self.rgb_uncertainty_out = nn.Linear(
                self.mlp_rgb_uncertainty.get_out_dim(), 3
            )

        if self.config.use_semantic_uncertainty:
            self.mlp_semantic_uncertainty = MLP(
                in_dim=self.config.geo_feat_dim,
                layer_width=self.config.semantic_uncertainty_mlp_width,
                num_layers=self.config.semantic_uncertainty_mlp_layers,
                activation=nn.ReLU(),
                out_activation=None,
            )
            self.semantic_uncertainty_out = nn.Linear(
                self.mlp_semantic_uncertainty.get_out_dim(), self.num_semantic_classes
            )

    def get_outputs(
        self,
        ray_samples: RaySamples,
        density_embedding: Optional[Tensor] = None,
        return_alphas: bool = False,
    ) -> Dict[FieldHeadNames, Tensor]:
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")

        outputs = {}

        camera_indices = ray_samples.camera_indices.squeeze()

        inputs = ray_samples.frustums.get_start_positions()
        inputs = inputs.view(-1, 3)

        directions = ray_samples.frustums.directions
        directions_flat = directions.reshape(-1, 3)

        inputs.requires_grad_(True)
        with torch.enable_grad():
            hidden_output = self.forward_geonetwork(inputs)
            sdf, geo_feature = torch.split(
                hidden_output, [1, self.config.geo_feat_dim], dim=-1
            )
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=inputs,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        rgb = self.get_colors(
            inputs, directions_flat, gradients, geo_feature, camera_indices
        )

        semantics_input = geo_feature.view(-1, self.config.geo_feat_dim)
        semantics_output = self.mlp_semantic(semantics_input)
        semantics = self.field_head_semantic(semantics_output)

        rgb_logvar = None
        if self.config.use_rgb_uncertainty:
            u_feat = self.mlp_rgb_uncertainty(semantics_input)
            raw_rgb_logvar = self.rgb_uncertainty_out(u_feat)
            rgb_logvar = torch.log(
                self.config.rgb_beta_min + torch.exp(raw_rgb_logvar)
            )

        semantic_logvar = None
        if self.config.use_semantic_uncertainty:
            su_feat = self.mlp_semantic_uncertainty(semantics_input)
            raw_semantic_logvar = self.semantic_uncertainty_out(su_feat)
            semantic_logvar = torch.log(
                self.config.semantic_beta_min + torch.exp(raw_semantic_logvar)
            )

        sample_shape = ray_samples.frustums.directions.shape[:-1]
        rgb = rgb.view(*sample_shape, -1)
        sdf = sdf.view(*sample_shape, -1)
        gradients = gradients.view(*sample_shape, -1)
        normals = torch.nn.functional.normalize(gradients, p=2, dim=-1)
        semantics = semantics.view(*sample_shape, -1)
        if rgb_logvar is not None:
            rgb_logvar = rgb_logvar.view(*sample_shape, -1)
        if semantic_logvar is not None:
            semantic_logvar = semantic_logvar.view(*sample_shape, -1)

        outputs.update(
            {
                FieldHeadNames.RGB: rgb,
                FieldHeadNames.SDF: sdf,
                FieldHeadNames.NORMALS: normals,
                FieldHeadNames.GRADIENT: gradients,
            }
        )
        if self.config.use_semantics:
            outputs[FieldHeadNames.SEMANTICS] = semantics
        if rgb_logvar is not None:
            outputs["rgb_uncertainty_logvar"] = rgb_logvar
        if semantic_logvar is not None:
            outputs["semantic_uncertainty_logvar"] = semantic_logvar

        if return_alphas:
            alphas = self.get_alpha(ray_samples, sdf, gradients)
            outputs.update({FieldHeadNames.ALPHA: alphas})

        return outputs
