import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tyro

sys.path.append(".")
from nerfstudio.scripts.train import main
from buildnet3d.models.method_configs import (
    NeRFactoTrackConfig,
    SplatfactoTrackConfig,
    NeuSTrackConfig,
    SemanticSDFTrackConfig,
    UncertaintySemanticSDFTrackConfig,
)


@dataclass
class TrainingConfig:
    """Configuration for training"""

    model_type: str = "uncertainty-semantic-sdf"
    """What NeRF model to train. Defaults to NeuS"""
    experiment_name: str = "test"
    """Name of the model to train"""
    output_dir: Path = "./outputs"
    """Where to save the model and outputs"""
    max_num_iterations: int = 100001
    data: Path = "buildnet3d/data/REAL/EXP1"
    """Where to find the input data """
    load_config: Optional[Path] = None
    load_dir: Optional[Path] = None
    # Path("outputs/2.1-0.1-semantic-uncertainty/semantic-sdf/2026-01-10_213757/nerfstudio_models")

    use_appearance_embedding: Optional[bool] = False
    use_average_appearance_embedding: Optional[bool] = True
    use_grid_feature: Optional[bool] = True

    # Model / datamanager overrides (used by experiments/*.yaml sweeps)
    fg_loss_relat_mult: Optional[float] = None
    rgb_uncertainty_loss_mult: Optional[float] = None
    semantic_uncertainty_loss_mult: Optional[float] = None
    semantic_loss_mult: Optional[float] = None
    use_rgb_uncertainty: Optional[bool] = None
    use_semantic_uncertainty: Optional[bool] = None
    rgb_uncertainty_delay_steps: Optional[int] = None
    semantic_uncertainty_delay_steps: Optional[int] = None
    N_reparam_samples: Optional[int] = None
    train_num_rays_per_batch: Optional[int] = None

    def __post_init__(self) -> None:
        method_config = {
            "nerfacto": NeRFactoTrackConfig,
            "neus": NeuSTrackConfig,
            "semantic-sdf": SemanticSDFTrackConfig,
            "uncertainty-semantic-sdf": UncertaintySemanticSDFTrackConfig,
            "splatfacto": SplatfactoTrackConfig,
        }
        if self.model_type not in method_config:
            raise ValueError(f"Model type {self.model_type} not supported.")
        # Deep-copy so CLI overrides do not mutate the shared module-level config.
        self.model = copy.deepcopy(method_config[self.model_type])
        self.model.experiment_name = self.experiment_name
        self.model.output_dir = self.output_dir
        self.model.max_num_iterations = self.max_num_iterations
        self.model.data = self.data
        self.model.load_config = self.load_config
        self.model.load_dir = self.load_dir

        model_cfg = self.model.pipeline.model
        if self.fg_loss_relat_mult is not None:
            model_cfg.fg_loss_relat_mult = self.fg_loss_relat_mult
        if self.rgb_uncertainty_loss_mult is not None:
            model_cfg.rgb_uncertainty_loss_mult = self.rgb_uncertainty_loss_mult
        if self.semantic_uncertainty_loss_mult is not None:
            model_cfg.semantic_uncertainty_loss_mult = self.semantic_uncertainty_loss_mult
        if self.semantic_loss_mult is not None:
            model_cfg.semantic_loss_mult = self.semantic_loss_mult
        if self.use_rgb_uncertainty is not None:
            model_cfg.use_rgb_uncertainty = self.use_rgb_uncertainty
        if self.use_semantic_uncertainty is not None:
            model_cfg.use_semantic_uncertainty = self.use_semantic_uncertainty
        if self.rgb_uncertainty_delay_steps is not None:
            model_cfg.rgb_uncertainty_delay_steps = self.rgb_uncertainty_delay_steps
        if self.semantic_uncertainty_delay_steps is not None:
            model_cfg.semantic_uncertainty_delay_steps = (
                self.semantic_uncertainty_delay_steps
            )
        if self.N_reparam_samples is not None:
            model_cfg.N_reparam_samples = self.N_reparam_samples
        if self.train_num_rays_per_batch is not None:
            self.model.pipeline.datamanager.train_num_rays_per_batch = (
                self.train_num_rays_per_batch
            )


if __name__ == "__main__":
    config = tyro.cli(TrainingConfig)

    main(config.model)
