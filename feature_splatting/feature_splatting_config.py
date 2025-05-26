from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

# Gsplat config templates
from feature_splatting.feature_splatting_datamgr import FeatureSplattingDataManagerConfig
from feature_splatting.model import FeatureSplattingModelConfig

from typing import Type
import torch
from torch.optim.optimizer import _use_grad_for_differentiable

from .config import TOTAL_STEPS, OPTIMIZER_SWITCH_STEP

class DelayedAdam(torch.optim.Adam):
    def __init__(self, n_min, **kwargs):
        super().__init__(**kwargs)
        self.i_steps = 0
        self.n_min = n_min
        
    @_use_grad_for_differentiable
    def step(self, closure=None):
        self.i_steps += 1
        if self.i_steps < self.n_min:
            return
        super().step(closure=closure)
        
class InterruptedAdam(torch.optim.Adam):
    def __init__(self, n_max, **kwargs):
        super().__init__(**kwargs)
        self.i_steps = 0
        self.n_max = n_max
        
    @_use_grad_for_differentiable
    def step(self, closure=None):
        self.i_steps += 1
        if self.i_steps > self.n_max:
            return
        super().step(closure=closure)
        
    
class DelayedAdamOptimizerConfig(AdamOptimizerConfig):
    """Basic optimizer config with Adam"""
    _target: Type = DelayedAdam
    weight_decay: float = 0
    """The weight decay to use."""
    
    # rude override of parent function from nerfstudio.engine.optimizers.OptimizerConfig
    def setup(self, params) -> torch.optim.Optimizer:
        """Returns the instantiated object using the config."""
        kwargs = vars(self).copy()
        kwargs.pop("_target")
        kwargs.pop("max_norm")
        kwargs["params"] = params
        # n_max SHOULD BE HARD_CODED HERE!
        self._target = DelayedAdam(n_min=OPTIMIZER_SWITCH_STEP, **kwargs)
        return self._target
    
class InterruptedAdamOptimizerConfig(AdamOptimizerConfig):
    """Basic optimizer config with Adam"""
    _target: Type = DelayedAdam
    weight_decay: float = 0
    """The weight decay to use."""
    
    # rude override of parent function from nerfstudio.engine.optimizers.OptimizerConfig
    def setup(self, params) -> torch.optim.Optimizer:
        """Returns the instantiated object using the config."""
        kwargs = vars(self).copy()
        kwargs.pop("_target")
        kwargs.pop("max_norm")
        kwargs["params"] = params
        # n_max SHOULD BE HARD_CODED HERE!
        self._target = InterruptedAdam(n_max=OPTIMIZER_SWITCH_STEP, **kwargs)
        return self._target

# Trainer config is modified from the template at
# https://github.com/nerfstudio-project/nerfstudio/blob/bf3664a19a89a61bcac83a9f69cbe2d6dc7c444d/nerfstudio/configs/method_configs.py#L594
feature_splatting_method = MethodSpecification(
    config=TrainerConfig(
        method_name="feature-splatting",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=TOTAL_STEPS,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=FeatureSplattingDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(load_3D_points=True),
                cache_images_type="uint8",
                enable_cache=False # changed to False to skip fature-saving (time and space consuming procedure)
            ),
            model=FeatureSplattingModelConfig(sh_degree=0),
        ),
        
        optimizers={
            "means": {
                "optimizer": InterruptedAdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=OPTIMIZER_SWITCH_STEP,
                ),
            },
            "features_dc": {
                "optimizer": InterruptedAdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": InterruptedAdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": InterruptedAdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": InterruptedAdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "distill_features": {
                "optimizer": DelayedAdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
                # "scheduler": ExponentialDecaySchedulerConfig(
                #     lr_final=5e-4,
                #     max_steps=10000,
                # ),
            },
            "feature_mlp": {
                "optimizer": DelayedAdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },
            "quats": {"optimizer": InterruptedAdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
            "camera_opt": {
                "optimizer": InterruptedAdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-7, max_steps=OPTIMIZER_SWITCH_STEP, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Feature Splatting distills language-aligned features into 3D Gaussians.",
)
