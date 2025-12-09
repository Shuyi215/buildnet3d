"""
Field for semantic-SDF, rather then estimating density to generate a surface,
a signed distance function (SDF) for surface representation is used to help with
extracting high fidelity surfaces, in addition with a 3D semantic segmentation.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Type

import torch
from jaxtyping import Float
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames, SemanticFieldHead
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.sdf_field import SDFField, SDFFieldConfig
from torch import Tensor, nn


@dataclass
class SemanticSDFFieldConfig(SDFFieldConfig):
    """Semantic-SDF Field Config

    Added RGB uncertainty head configuration so that we can model
    heteroscedastic (per-ray) photometric uncertainty.
    """

    _target: Type = field(default_factory=lambda: SemanticSDFField)
    # Semantic settings
    num_semantic_classes: int = 6
    semantic_mlp_number_layers: int = 2
    semantic_mlp_layer_width: int = 128
    """Numbers of neurons in one layer of the semantic MLP"""
    
    # RGB uncertainty settings
    use_rgb_uncertainty: bool = True
    """Whether to predict per-sample RGB log-variance (heteroscedastic uncertainty)."""
    rgb_uncertainty_mlp_layers: int = 2
    """Number of hidden layers for RGB uncertainty MLP."""
    rgb_uncertainty_mlp_width: int = 64
    """Width of hidden layers for RGB uncertainty MLP."""
    rgb_beta_min: float = 0.001
    """Minimum beta value for RGB uncertainty log value."""
    
    # semantic uncertainty settings
    use_semantic_uncertainty: bool = True
    """Whether to predict per-sample semantic log-variance (heteroscedastic uncertainty)."""
    semantic_uncertainty_mlp_layers: int = 2
    """Number of hidden layers for semantic uncertainty MLP."""
    semantic_uncertainty_mlp_width: int = 64
    """Width of hidden layers for semantic uncertainty MLP."""
    semantic_beta_min: float = 0.001
    """Minimum beta value for semantic uncertainty log value."""

class SemanticSDFField(SDFField):
    """
    A field that learns a Signed Distance Functions (SDF), an RGB color and a
    semantic segmentation.
    """

    config: SemanticSDFFieldConfig

    def __init__(
        self,
        config: SemanticSDFFieldConfig,
        aabb: Float[Tensor, "2 3"],  # noqa: F722
        num_images: int,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
    ) -> None:
        """
        The field is built on top of the SDF field. It needs a `config` file, the scene
        size as an axis-aligned bounding box in `aabb` and the number of images used for
        embedding appearance in `num_images`. Set `use_average_appearance_embedding` if
        necessary and specify the `spatial_distortion` if there is some.
        """
        super().__init__(
            config,
            aabb,
            num_images,
            use_average_appearance_embedding,
            spatial_distortion,
        )

        self.num_semantic_classes = self.config.num_semantic_classes
        self.mlp_semantic = MLP(
            in_dim=self.config.geo_feat_dim,
            layer_width=self.config.semantic_mlp_layer_width,
            num_layers=self.config.semantic_mlp_number_layers,
            activation=nn.ReLU(),
            out_activation=nn.ReLU(),
        )
        self.field_head_semantic = SemanticFieldHead(
            in_dim=self.mlp_semantic.get_out_dim(),
            num_classes=self.num_semantic_classes,
        )

        # RGB uncertainty head (predict log-variance for each RGB channel)
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
            )  # 3-channel log variance (per RGB)
            
        # Semantic uncertainty head (predict log-variance for each semantic class)
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
            )  # per-class log variance

    def get_outputs(
        self,
        ray_samples: RaySamples,
        density_embedding: Optional[Tensor] = None,
        return_alphas: bool = False,
    ) -> Dict[FieldHeadNames, Tensor]:
        """
        Compute output of the field using the `ray_samples` as input.
        `density_embedding` is a useless artifact from base class. Use
        `return_aphas` if needed
        """
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

        # RGB uncertainty (per sample) - predict log variance so it's unconstrained
        if self.config.use_rgb_uncertainty:
            uncertainty_input = geo_feature.view(-1, self.config.geo_feat_dim)
            u_feat = self.mlp_rgb_uncertainty(uncertainty_input)
            raw_rgb_logvar = self.rgb_uncertainty_out(u_feat)
            rgb_logvar = torch.log(self.config.rgb_beta_min + torch.exp(raw_rgb_logvar))
        else:
            rgb_logvar = None
            
        # Semantic uncertainty (per sample) - predict log variance so it's unconstrained
        if self.config.use_semantic_uncertainty:
            sem_uncertainty_input = geo_feature.view(-1, self.config.geo_feat_dim)
            su_feat = self.mlp_semantic_uncertainty(sem_uncertainty_input)
            raw_semantic_logvar = self.semantic_uncertainty_out(su_feat)
            semantic_logvar = torch.log(self.config.semantic_beta_min + torch.exp(raw_semantic_logvar))
        else:
            semantic_logvar = None

        rgb = rgb.view(*ray_samples.frustums.directions.shape[:-1], -1)
        sdf = sdf.view(*ray_samples.frustums.directions.shape[:-1], -1)
        gradients = gradients.view(*ray_samples.frustums.directions.shape[:-1], -1)
        normals = torch.nn.functional.normalize(gradients, p=2, dim=-1)
        semantics = semantics.view(*ray_samples.frustums.directions.shape[:-1], -1)
        if rgb_logvar is not None:
            rgb_logvar = rgb_logvar.view(*ray_samples.frustums.directions.shape[:-1], -1)
        if semantic_logvar is not None:
            semantic_logvar = semantic_logvar.view(*ray_samples.frustums.directions.shape[:-1], -1)

        outputs.update(
            {
                FieldHeadNames.RGB: rgb,
                FieldHeadNames.SDF: sdf,
                FieldHeadNames.NORMALS: normals,
                FieldHeadNames.GRADIENT: gradients,
                FieldHeadNames.SEMANTICS: semantics,
            }
        )
        if rgb_logvar is not None:
            outputs["rgb_uncertainty"] = rgb_logvar
        if semantic_logvar is not None:
            outputs["semantic_uncertainty"] = semantic_logvar

        if return_alphas:
            alphas = self.get_alpha(ray_samples, sdf, gradients)
            outputs.update({FieldHeadNames.ALPHA: alphas})

        return outputs
