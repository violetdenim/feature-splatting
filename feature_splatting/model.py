import os, time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union
import torchvision.transforms as transforms

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig, get_viewmat, resize_image
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.viewer.server.viewer_elements import ViewerButton

from nerfstudio.data.scene_box import OrientedBox
from .config import TOTAL_STEPS, OPTIMIZER_SWITCH_STEP, USE_2DGS

# Feature splatting functions
from torch.nn import Parameter
from feature_splatting.utils import (
    ViewerUtils,
    apply_pca_colormap_return_proj,
    # two_layer_mlp,
    cnn_decoder
)
try:
    # from gsplat.cuda._torch_impl import _quat_to_rotmat
    from gsplat.rendering import rasterization, rasterization_2dgs
except ImportError:
    print("Please install gsplat>=1.0.0")
    
    
    
    
def contrastive_2d_loss(segmask, features, dim_features):
    """
    Compute the contrastive clustering loss for a 2D feature map.

    :param segmask: Tensor of shape (H, W).
    :param features: Tensor of shape (H, W, D), where (H, W) is the resolution and D is the dimensionality of these features.
    :param id_unique_list: Tensor of shape (n_p).
    :n_i_list: Tensor of shape (n_p).
    :dim_features: is the dimensionality of the features (equal to D).
    :lambda_val: Weighting factor for the loss.

    :return: loss value.
    """

    # get number of masks and number of px per mask
    segmask = segmask.squeeze(-1)
    id_unique_list, n_i_list = torch.unique(segmask, return_counts=True)
    
    n_p = id_unique_list.shape[0] # Number of ids
    
    features = features / (features.norm(dim=-1, keepdim=True) + 1e-9)

    f_mean_per_cluster = torch.zeros((n_p, dim_features)).cuda()
    phi_per_cluster = torch.zeros((n_p, 1)).cuda()

    # go over all masks
    # f_is = []
    for i in range(n_p):
        # get all features, corresponding to current mask and take the mean
        f_i = features[(segmask == id_unique_list[i]).to(segmask.device), :]
        # f_is.append(f_i)
        
        f_mean_per_cluster[i, ...] = torch.mean(f_i, dim=0, keepdim=True)
        
        # temperature for softmax, calculated as average distance between cluster mean and all features
        phi_per_cluster[i] = torch.norm(f_i - f_mean_per_cluster[i], dim=1, keepdim=True).sum() / (n_i_list[i] * torch.log(n_i_list[i] + 100))
            
    phi_per_cluster = torch.clip(phi_per_cluster * 10, min=0.1, max=1.0)
    phi_per_cluster = phi_per_cluster.detach()
    loss_per_cluster = torch.zeros(n_p).cuda()

    for i in range(n_p):
        f_mean = f_mean_per_cluster[i]
        phi = phi_per_cluster[i]
        f_ij = features[(segmask == id_unique_list[i]).to(segmask.device), :]
        
        num = torch.exp(torch.matmul(f_ij, f_mean) / (phi + 1e-6)) # dim (ni)
        den = torch.sum(torch.exp(torch.matmul(f_ij, f_mean_per_cluster.transpose(-1, -2)) / (phi_per_cluster.transpose(-1, -2) + 1e-6)), dim=1) # dim (n_i)
        
        loss_per_cluster[i] = torch.sum(torch.log(num / (den + 1e-6)))
            
    return -torch.mean(loss_per_cluster)


@dataclass
class FeatureSplattingModelConfig(SplatfactoModelConfig):
    """Note: make sure to use naming that doesn't conflict with NerfactoModelConfig"""

    _target: Type = field(default_factory=lambda: FeatureSplattingModel)
    # Compute SHs in python
    python_compute_sh: bool = False
    # Weighing for the overall feature loss
    feat_loss_weight: float = 1e-3
    feat_aux_loss_weight: float = 0.1
    # Latent dimension for the feature field
    # TODO(roger): this feat_dim has to add up depth/color to a number that can be rasterized without padding
    # https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/cuda/_wrapper.py#L431
    # gsplat's N-D implementation seems to have some bugs that cause padded tensors to have memory issues
    # we can create a PR to fix this.
    feat_latent_dim: int = 13
    # Feature Field MLP Head
    mlp_hidden_dim: int = 64

def cosine_loss(network_output, gt):
    assert network_output.shape == gt.shape
    return (1 - F.cosine_similarity(network_output, gt, dim=0)).mean()

class FeatureSplattingModel(SplatfactoModel):
    config: FeatureSplattingModelConfig

    def populate_modules(self):
        self.i_iteration = 0
        super().populate_modules()
        # Sanity check
        if self.config.python_compute_sh:
            raise NotImplementedError("Not implemented yet")
        if self.config.sh_degree > 0:
            assert self.config.python_compute_sh, "SH computation is only supported in python"
        else:
            assert not self.config.python_compute_sh, "SHs python compute flag should not be used with 0 SH degree"
        
        # Initialize per-Gaussian features
        distill_features = torch.nn.Parameter(torch.zeros((self.means.shape[0], self.config.feat_latent_dim)))
        self.gauss_params["distill_features"] = distill_features
        
        self.main_feature_name = self.kwargs["metadata"]["main_feature_name"]
        self.main_feature_shape_chw = self.kwargs["metadata"]["feature_dim_dict"][self.main_feature_name]

        # Initialize the multi-head feature MLP
        self.feature_mlp = cnn_decoder(self.config.feat_latent_dim,
                                         #self.config.mlp_hidden_dim,
                                         self.kwargs["metadata"]["feature_dim_dict"])
        
        # Visualization utils
        self.maybe_populate_text_encoder()
        self.setup_gui()
    
    def maybe_populate_text_encoder(self):
        self.text_encoding_func = lambda x: x
    
    def setup_gui(self):
        self.viewer_utils = ViewerUtils(self.text_encoding_func)
        # Note: the GUI elements are shown based on alphabetical variable names
        self.btn_refresh_pca = ViewerButton("Refresh PCA Projection", cb_hook=lambda _: self.viewer_utils.reset_pca_proj())

    def physics_sim_step(self):
        # It's just a placeholder now. NS needs some user interaction to send rendering requests.
        # So I make a button that does nothing but to trigger rendering.
        pass
    
    def estimate_ground(self):
        pass
    
    def start_editing(self):
        pass
    def segment_positive_obj(self):
        pass
    
    def segment_gaussian(self, field_name : str, use_canonical : bool, sample_size : Optional[int] = 2**15, threshold : Optional[float] = 0.5):
        return None, None

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        # cropping
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(
                    int(camera.width.item()), int(camera.height.item()), self.background_color
                )
        else:
            crop_ids = None

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
            distill_features_crop = self.distill_features[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats
            distill_features_crop = self.distill_features

        # features_dc_crop.shape: [N, 3]
        # features_rest_crop.shape: [N, 15, 3]
        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)
        
        # colors_crop.shape: [N, 16, 3]

        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default
        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
       
        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        if self.config.output_depth_during_training or not self.training:
            # Actually render RGB, features, and depth, but can't use RGB+FEAT+ED because we hack gsplat
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"
            
        

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
            assert self.config.python_compute_sh, "SH computation is only supported in python"
            raise NotImplementedError("Python SHs computation not implemented yet")
            sh_degree_to_use = None
        else:
            colors_crop = torch.sigmoid(colors_crop).squeeze(1)  # [N, 1, 3] -> [N, 3]
            fused_render_properties = torch.cat((colors_crop, distill_features_crop), dim=1)
            sh_degree_to_use = None
        
        if not USE_2DGS:
            render, alpha, self.info = rasterization(
                means=means_crop,
                quats=quats_crop / quats_crop.norm(dim=-1, keepdim=True),
                scales=torch.exp(scales_crop),
                opacities=torch.sigmoid(opacities_crop).squeeze(-1),
                colors=fused_render_properties,
                viewmats=viewmat,  # [1, 4, 4]
                Ks=K,  # [1, 3, 3]
                width=W,
                height=H,
                tile_size=BLOCK_WIDTH,
                packed=False,
                near_plane=0.01,
                far_plane=1e10,
                render_mode=render_mode,
                sh_degree=sh_degree_to_use,
                sparse_grad=False,
                absgrad=True,
                rasterize_mode=self.config.rasterize_mode,
                # set some threshold to disregrad small gaussians for faster rendering.
                # radius_clip=3.0,
            )
            # ignore during error computation
            normals1 = torch.tensor((1.0, 0.0, 0.0))
            normals2 = normals1
        else:
            render_mode = "RGB+ED"
            render, alpha, normals, normals_from_depth, render_distort, render_median, self.info = rasterization_2dgs(
                means=means_crop,
                quats=quats_crop / quats_crop.norm(dim=-1, keepdim=True),
                scales=torch.exp(scales_crop),
                opacities=torch.sigmoid(opacities_crop).squeeze(-1),
                colors=fused_render_properties,
                viewmats=viewmat,  # [1, 4, 4]
                Ks=K,  # [1, 3, 3]
                width=W,
                height=H,
                tile_size=BLOCK_WIDTH,
                packed=False,
                near_plane=0.01,
                far_plane=1e10,
                render_mode=render_mode,
                sh_degree=sh_degree_to_use,
                sparse_grad=False,
                absgrad=True,
                depth_mode="expected" # "median"
            )
            normals1 = normals.squeeze(0) 
            normals2 = normals_from_depth.squeeze(0)
            
        if self.training and self.info["means2d"].requires_grad:
            self.info["means2d"].retain_grad()
        self.xys = self.info["means2d"]  # [1, N, 2]
        self.radii = self.info["radii"][0]  # [N]
        alpha = alpha[:, ...]

        background = self._get_background_color()
        rgb = render[:, ..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)

        if render_mode == "RGB+ED":
            assert render.shape[3] == 3 + self.config.feat_latent_dim + 1
            depth_im = render[:, ..., -1:]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
        else:
            assert render.shape[3] == 3 + self.config.feat_latent_dim
            depth_im = None

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)
        
        feature = render[:, ..., 3:3 + self.config.feat_latent_dim]
        
        return {
            "rgb": rgb.squeeze(0),  # type: ignore
            "depth": depth_im,  # type: ignore
            
            "normals": normals1, 
            "normals_from_depth": normals2, 
            
            "accumulation": alpha.squeeze(0),  # type: ignore
            "background": background,  # type: ignore,
            "feature": feature.squeeze(0),  # type: ignore
        }  # type: ignore
        

    def decode_features(self, features_hwc: torch.Tensor, resize_factor: float = 1.) -> Dict[str, torch.Tensor]:
        # Decode features
        feature_chw = features_hwc.permute(2, 0, 1)
        feature_shape_hw = (int(self.main_feature_shape_chw[1] * resize_factor), int(self.main_feature_shape_chw[2] * resize_factor))
        rendered_feat = F.interpolate(feature_chw.unsqueeze(0), size=feature_shape_hw, mode="bilinear", align_corners=False)
        rendered_feat_dict = self.feature_mlp(rendered_feat)
        # Rest of the features
        for key, feat_shape_chw in self.kwargs["metadata"]["feature_dim_dict"].items():
            if key != self.main_feature_name:
                rendered_feat_dict[key] = F.interpolate(rendered_feat_dict[key], size=feat_shape_chw[1:], mode="bilinear", align_corners=False)
            rendered_feat_dict[key] = rendered_feat_dict[key].squeeze(0)
        return rendered_feat_dict
    
    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        # print("FeatureSplattingModel::get_loss_dict")
        # Splatfacto computes the loss for the rgb image
        
        # loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        
        scale_reg = torch.tensor(1.0).to(self.device)
        normal_loss = torch.tensor(0.0).to(self.device)
        img_loss = torch.tensor(0.0).to(self.device)
        feature_loss = torch.tensor(0.0).to(self.device)
        
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["rgb"]

        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])
        img_loss = (1 - self.config.ssim_lambda) * torch.abs(gt_img - pred_img).mean() + self.config.ssim_lambda * simloss 
        
        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.scales)
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio),
                )
                - self.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        
        if self.i_iteration > 7000:
            normal_error = (1 - (outputs['normals'] * outputs['normals_from_depth']).sum(dim=0))[None]
            normal_loss = 0.05 * (normal_error).mean()
        
        if self.i_iteration >= OPTIMIZER_SWITCH_STEP:
            target_feat = batch['feature_dict']['dinov2'].to(self.device)
            decoded_feature = self.decode_features(outputs["feature"])["dinov2"]
            
            ignore_feat_mask = (torch.sum(target_feat == 0, dim=0) == target_feat.shape[0])
            target_feat[:, ignore_feat_mask] = decoded_feature[:, ignore_feat_mask]
            # L1
            # feature_loss = self.config.feat_loss_weight * torch.abs(decoded_feature - target_feat).mean() 
            
            decoded_feature = decoded_feature.permute(1, 2, 0)
            assert("segmentation" in batch)
            sg_mask = batch["segmentation"]
            assert sg_mask.shape[:2] == decoded_feature.shape[:2]
            assert sg_mask.ndim == 2
            assert decoded_feature.ndim == 3
            assert decoded_feature.shape[-1] == 384
            feature_loss = 1e-4 * contrastive_2d_loss(sg_mask.long().to(self.device),
                                                      decoded_feature,
                                                      dim_features=decoded_feature.shape[-1])
        
        loss_dict = {
            "main_loss": img_loss + normal_loss, 
            "feature_loss": feature_loss,
            "scale_reg": scale_reg,
        }
        
        if self.training:
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)
        
        self.i_iteration += 1
        # print(loss_dict)
        
        return loss_dict
    
    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """This function is not called during training, but used for visualization in browser. So we can use it to
        add visualization not needed during training.
        """
        
        # outs = super().get_outputs_for_camera(camera, obb_box)
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)
        outs = self.get_outputs(camera.to(self.device))
        outs["consistent_latent_pca"], self.viewer_utils.pca_proj, *_ = apply_pca_colormap_return_proj(
            outs["feature"], self.viewer_utils.pca_proj
        )
        
        return outs
    
    # ===== Utils functions for managing the gaussians =====

    @property
    def distill_features(self):
        return self.gauss_params["distill_features"]
    
    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        self.step = 30000
        if "means" in dict:
            # For backwards compatibility, we remap the names of parameters from
            # means->gauss_params.means since old checkpoints have that format
            for p in self.gauss_params.keys():
                dict[f"gauss_params.{p}"] = dict[p]
        newp = dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            old_shape = param.shape
            new_shape = (newp,) + old_shape[1:]
            self.gauss_params[name] = torch.nn.Parameter(torch.zeros(new_shape, device=self.device))
        super().load_state_dict(dict, **kwargs)
    
    def split_gaussians(self, split_mask, samps):
        """
        This function splits gaussians that are too large
        """
        n_splits = split_mask.sum().item()
        CONSOLE.log(f"Splitting {split_mask.sum().item()/self.num_points} gaussians: {n_splits}/{self.num_points}")
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
            torch.exp(self.scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        quats = self.quats[split_mask] / self.quats[split_mask].norm(dim=-1, keepdim=True)  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self.means[split_mask].repeat(samps, 1)
        # step 2, sample new colors
        new_features_dc = self.features_dc[split_mask].repeat(samps, 1)
        new_features_rest = self.features_rest[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_opacities = self.opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self.scales[split_mask]) / size_fac).repeat(samps, 1)
        self.scales[split_mask] = torch.log(torch.exp(self.scales[split_mask]) / size_fac)
        # step 5, sample new quats
        new_quats = self.quats[split_mask].repeat(samps, 1)
        # step 6 (RQ, July 2024), sample new distill_features
        new_distill_features = self.distill_features[split_mask].repeat(samps, 1)
        out = {
            "means": new_means,
            "features_dc": new_features_dc,
            "features_rest": new_features_rest,
            "opacities": new_opacities,
            "scales": new_scales,
            "quats": new_quats,
            "distill_features": new_distill_features,
        }
        for name, param in self.gauss_params.items():
            if name not in out:
                out[name] = param[split_mask].repeat(samps, 1)
        return out

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        return {
            name: [self.gauss_params[name]]
            for name in self.gauss_params.keys()
        }
    
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        # Gather Gaussian-related parameters
        # The distill_features parameter is added via the get_gaussian_param_groups method
        param_groups = super().get_param_groups()
        param_groups["feature_mlp"] = list(self.feature_mlp.parameters())
        return param_groups
    
    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        metrics_dict = {}
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        metrics_dict["gaussian_count"] = self.num_points

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict
