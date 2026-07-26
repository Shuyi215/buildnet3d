# model.configs.py

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig
)
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
    CosineDecaySchedulerConfig
    )

from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.models.neus import NeuSModelConfig
from nerfstudio.fields.sdf_field import SDFFieldConfig
from nerfstudio.models.splatfacto import SplatfactoModelConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from buildnet3d.data.buildnet_dataset import BuildNetDataset
from buildnet3d.data.buildnet_dataparser import BuildNetDataParserConfig

from buildnet3d.models.semantic_sdf import SemanticSDFModelConfig
from buildnet3d.models.semantic_sdf_field import SemanticSDFFieldConfig
from buildnet3d.models.uncertainty_semantic_sdf import UncertaintySemanticSDFModelConfig
from buildnet3d.models.uncertainty_semantic_sdf_field import (
    UncertaintySemanticSDFFieldConfig,
)

NeuSTrackConfig = TrainerConfig(
    method_name="neus",
    steps_per_eval_image=1500,
    steps_per_eval_batch=500,
    steps_per_save=5000,
    steps_per_eval_all_images=1000000,
    max_num_iterations=100001,
    save_only_latest_checkpoint= False,
    mixed_precision=False,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            _target=VanillaDataManager[BuildNetDataset],
            dataparser=BuildNetDataParserConfig(
                meta_json_filename="meta_data.json",
                meta_rgb_path="images",
            ),
            train_num_rays_per_batch=512,
            eval_num_rays_per_batch=256,
        ),
        model=NeuSModelConfig(
            sdf_field=SDFFieldConfig(
                bias=0.5,
                beta_init=0.2,
                inside_outside=False,
                use_grid_feature=True,
                use_appearance_embedding=False,
            ),
            near_plane=0.05,
            far_plane=50.0,
            background_model="none",
            background_color="random",
            overwrite_near_far_plane=False,
            use_average_appearance_embedding=False,
            eval_num_rays_per_chunk=256,
            eikonal_loss_mult=0.1,
        ),
    ),
    optimizers={
        "camera_opt": {
            "mode": "off",
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=10000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": CosineDecaySchedulerConfig(
                warm_up_end=5000, learning_rate_alpha=0.01, max_steps=300001
            ),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": CosineDecaySchedulerConfig(
                warm_up_end=5000, learning_rate_alpha=0.01, max_steps=300001
            ),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="wandb",
)

NeRFactoTrackConfig = TrainerConfig(
    method_name="nerfacto",
    steps_per_eval_image=1000,
    steps_per_eval_batch=1000,
    steps_per_save=5000,
    max_num_iterations=50001,
    save_only_latest_checkpoint= False,
    mixed_precision=False,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=BuildNetDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
        ),
        model=NerfactoModelConfig(
            eval_num_rays_per_chunk=1 << 15,
            average_init_density=0.01,
            background_color="random",
            camera_optimizer=CameraOptimizerConfig(mode="off"),
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=100000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=100000),
        },
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=10000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15, quit_on_train_completion=True),
    vis="wandb",
)

SplatfactoTrackConfig = TrainerConfig(
    method_name="splatfacto",
    steps_per_eval_image=1000,
    steps_per_eval_batch=1000,
    steps_per_save=5000,
    max_num_iterations=50001,
    save_only_latest_checkpoint= False,
    mixed_precision=False,
    pipeline=VanillaPipelineConfig(
        datamanager=FullImageDatamanagerConfig(
            dataparser=BuildNetDataParserConfig(),
            cache_images_type="uint8",
        ),
        model=SplatfactoModelConfig(),
    ),
    optimizers={
        "means": {
            "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1.6e-6,
                max_steps=30000,
            ),
        },
        "features_dc": {
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        },
        "features_rest": {
            "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
            "scheduler": None,
        },
        "opacities": {
            "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
            "scheduler": None,
        },
        "scales": {
            "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
            "scheduler": None,
        },
        "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
        "camera_opt": {
            "mode": "off",
            "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=5e-7, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
            ),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15, quit_on_train_completion=True),
    vis="wandb",
)

SemanticSDFTrackConfig = TrainerConfig(
    method_name="semantic-sdf",
    steps_per_eval_image=1500,
    steps_per_eval_batch=500,
    steps_per_save=5000,
    steps_per_eval_all_images=1000000,
    max_num_iterations=100001,
    save_only_latest_checkpoint=True,
    mixed_precision=False,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            _target=VanillaDataManager[BuildNetDataset],
            dataparser=BuildNetDataParserConfig(
                meta_json_filename="meta_data.json",
                meta_rgb_path="images",
                split_by_filename=False,
                skip_every_for_val_split=8,  # n=8 → 每 8 张取 1 张做 eval
                semantics=True,
            ),
            train_num_rays_per_batch=512,
            eval_num_rays_per_batch=256,
        ),
        model=SemanticSDFModelConfig(
            # --- SDF field ---
            sdf_field=SemanticSDFFieldConfig(
                bias=0.5,
                beta_init=0.2,
                inside_outside=False,
                use_grid_feature=True,
                use_appearance_embedding=False,
                use_semantics=True,
                num_semantic_classes=5,
            ),
            # --- rendering ---
            near_plane=0.05,
            far_plane=50.0,
            overwrite_near_far_plane=False,
            background_model="none",
            background_color="random",
            use_average_appearance_embedding=False,
            eval_num_rays_per_chunk=256,
            # --- foreground ---
            fg_loss_relat_mult=1000,
            # --- semantics ---
            use_semantics=True,
            semantic_loss_mult=0.5,
        ),
    ),
    optimizers={
        "camera_opt": {
            "mode": "off",
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=10000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15, max_norm=1.0),
            "scheduler": CosineDecaySchedulerConfig(
                warm_up_end=5000, learning_rate_alpha=0.01, max_steps=300001
            ),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": CosineDecaySchedulerConfig(
                warm_up_end=5000, learning_rate_alpha=0.01, max_steps=300001
            ),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="wandb",
)

UncertaintySemanticSDFTrackConfig = TrainerConfig(
    method_name="uncertainty-semantic-sdf",
    steps_per_eval_image=1500,
    steps_per_eval_batch=500,
    steps_per_save=5000,
    steps_per_eval_all_images=1000000,
    max_num_iterations=100001,
    save_only_latest_checkpoint=True,
    mixed_precision=False,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            _target=VanillaDataManager[BuildNetDataset],
            dataparser=BuildNetDataParserConfig(
                meta_json_filename="meta_data.json",
                meta_rgb_path="images",
                split_by_filename=False,
                skip_every_for_val_split=8,
                semantics=True,
            ),
            train_num_rays_per_batch=512,
            eval_num_rays_per_batch=256,
        ),
        model=UncertaintySemanticSDFModelConfig(
            # --- SDF field ---
            sdf_field=UncertaintySemanticSDFFieldConfig(
                bias=0.5,
                beta_init=0.2,
                inside_outside=False,
                use_grid_feature=True,
                use_appearance_embedding=False,
                use_semantics=True,
                num_semantic_classes=5,
                # --- uncertainty ---
                use_rgb_uncertainty=True,
                rgb_beta_min=0.0001,
                use_semantic_uncertainty=True,
                semantic_beta_min=0.0001,
            ),
            # --- rendering ---
            near_plane=0.05,
            far_plane=50.0,
            overwrite_near_far_plane=False,
            background_model="none",
            background_color="random",
            use_average_appearance_embedding=False,
            eval_num_rays_per_chunk=256,
            # --- foreground ---
            fg_loss_relat_mult=1000,
            # --- semantics ---
            use_semantics=True,
            semantic_loss_mult=0.5,
            # --- rgb uncertainty ---
            use_rgb_uncertainty=True,
            rgb_uncertainty_delay_steps=5000,
            rgb_uncertainty_loss_mult=1.0,
            # --- semantic uncertainty ---
            use_semantic_uncertainty=True,
            semantic_uncertainty_delay_steps=10000,
            reparameterize_semantic_uncertainty=True,
            N_reparam_samples=10,
        ),
    ),
    optimizers={
        "camera_opt": {
            "mode": "off",
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=10000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15, max_norm=1.0),
            "scheduler": CosineDecaySchedulerConfig(
                warm_up_end=5000, learning_rate_alpha=0.01, max_steps=300001
            ),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": CosineDecaySchedulerConfig(
                warm_up_end=5000, learning_rate_alpha=0.01, max_steps=300001
            ),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="wandb",
)

semantic_sdf_method = MethodSpecification(
    config=SemanticSDFTrackConfig,
    description="Semantic SDF (NeuS + semantics)",
)

uncertainty_semantic_sdf_method = MethodSpecification(
    config=UncertaintySemanticSDFTrackConfig,
    description="Semantic SDF with RGB / semantic uncertainty",
)