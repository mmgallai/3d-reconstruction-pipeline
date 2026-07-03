"""MethodSpecification: splatfacto-tof = splatfacto-big + ToF depth loss.

Mirrors nerfstudio's splatfacto-big config (looser culling, no
densification cap) but with our depth-supervised model + a datamanager
that loads `depth_image` into the batch.
"""
from __future__ import annotations

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from splat_tof.depth_datamanager import FullImageDepthDatamanagerConfig
from splat_tof.model import SplatfactoTofModelConfig


splatfacto_tof = MethodSpecification(
    config=TrainerConfig(
        method_name="splatfacto-tof",
        steps_per_eval_image=500,
        steps_per_eval_batch=500,
        steps_per_save=2000,
        steps_per_eval_all_images=1000000,
        max_num_iterations=30000,
        mixed_precision=False,
        gradient_accumulation_steps={"camera_opt": 100, "color": 10, "shs": 10},
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDepthDatamanagerConfig(
                dataparser=NerfstudioDataParserConfig(
                    load_3D_points=True,
                    depth_unit_scale_factor=1.0,  # our .npy depths are float32 metres
                ),
                cache_images_type="uint8",
            ),
            model=SplatfactoTofModelConfig(
                # Match splatfacto-big density behaviour: looser culling, more
                # Gaussians retained. Defaults below mirror SplatfactoBigModelConfig
                # in nerfstudio (cull_alpha_thresh=0.005, stop_split_at=25k).
                cull_alpha_thresh=0.005,
                stop_split_at=25000,
                densify_grad_thresh=0.0006,
                # Depth-supervision knobs (defined in SplatfactoTofModelConfig).
                depth_lambda=0.02,
                depth_min=0.05,
                depth_max=10.0,
            ),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6, max_steps=30000,
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
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-5, max_steps=30000,
                ),
            },
            "bilateral_grid": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=2e-4, max_steps=30000,
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="tensorboard",
    ),
    description=("splatfacto-big + Femto ToF depth loss. Splatfacto's "
                 "densification stays active (unlike dn-splatter on gsplat 1.5), "
                 "but each iteration pulls Gaussians toward the measured depth."),
)
