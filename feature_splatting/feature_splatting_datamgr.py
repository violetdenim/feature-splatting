import gc
import os
from dataclasses import dataclass, field
from typing import Dict, Tuple, Type
from nerfstudio.cameras.cameras import Cameras

from tqdm import tqdm
import numpy as np
import torch
from jaxtyping import Float
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanager,
    FullImageDatamanagerConfig,
)
from nerfstudio.utils.rich_utils import CONSOLE

from feature_splatting.feature_extractor_cfg import DINOArgs
from feature_splatting.feature_extractor import batch_extract_feature
import cv2

@dataclass
class FeatureSplattingDataManagerConfig(FullImageDatamanagerConfig):
    _target: Type = field(default_factory=lambda: FeatureSplattingDataManager)
    """Feature type to extract."""
    enable_cache: bool = True
    """Whether to cache extracted features."""

class FeatureSplattingDataManager(FullImageDatamanager):
    config: FeatureSplattingDataManagerConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Extract features
        feature_dict = self.extract_features()

        # Split into train and eval features
        self.train_feature_dict = {}
        self.eval_feature_dict = {}
        feature_dim_dict = {}
        for feature_name in feature_dict:
            assert len(feature_dict[feature_name]) == len(self.train_dataset) + len(self.eval_dataset)
            feature_dim_dict[feature_name] = feature_dict[feature_name].shape[1:]  # c, h, w
            # split array in two parts
            self.train_feature_dict[feature_name], self.eval_feature_dict[feature_name] = feature_dict[feature_name][:len(self.train_dataset)], feature_dict[feature_name][len(self.train_dataset):]
            
        assert len(self.eval_feature_dict[feature_name]) == len(self.eval_dataset)

        del feature_dict
        # Garbage collect
        torch.cuda.empty_cache()
        gc.collect()
        
        # Set metadata, so we can initialize model with feature dimensionality
        self.train_dataset.metadata["feature_type"] = "DINO"
        self.train_dataset.metadata["feature_dim_dict"] = feature_dim_dict
        self.train_dataset.metadata["main_feature_name"] = "dinov2"
        
        # load segmentations
        print('Loading segmentations!')
        segm_path = os.path.join(self.config.data / "segmentations")
        assert(os.path.exists(segm_path))
        segm_files = sorted(os.listdir(segm_path))
        
        self.train_segment = []
        self.eval_segment = []
        for i_file, f in tqdm(enumerate(segm_files), total=len(segm_files)):
            segm_img = np.load(os.path.join(segm_path, f))
            # downsample immediately to feature shape
            h, w = feature_dim_dict[feature_name][-2:]
            segm_img = cv2.resize(segm_img, (w, h), interpolation=cv2.INTER_NEAREST)
            segm_img = torch.tensor(segm_img).to(self.device).squeeze(0)
            
            if i_file < len(self.train_dataset):
                self.train_segment.append(segm_img)
            else:
                self.eval_segment.append(segm_img)
                
        # this class will be used for rather fast upsampling
        self.upsampler = None
        # Garbage collect
        torch.cuda.empty_cache()
        gc.collect()
    
    def extract_features(self) -> Dict[str, Float[torch.Tensor, "n h w c"]]:
        # Extract features

        extract_args = DINOArgs
        image_fnames = self.train_dataset.image_filenames + self.eval_dataset.image_filenames
        # For dev purpose, visually tested image_fnames order matches camera_idx. NS seems to internally sort valid image_fnames.
        # self.feature_image_fnames = image_fnames

        # If cache exists, load it and validate it. We save it to the dataset directory.
        cache_dir = self.config.dataparser.data
        cache_path = cache_dir / f"feature_splatting_dino_features.pt"
        if self.config.enable_cache and os.path.exists(cache_path):
            cache_dict = torch.load(cache_dir)
            if cache_dict.get("image_fnames") != image_fnames:
                CONSOLE.print("Image filenames have changed, cache invalidated...")
            elif cache_dict.get("args") != extract_args.id_dict():
                CONSOLE.print("Feature extraction args have changed, cache invalidated...")
            else:
                return cache_dict["feature_dict"]
            

        # Cache is invalid or doesn't exist, so extract features
        CONSOLE.print(f"Extracting DINO features for {len(image_fnames)} images...")
        feature_dict = batch_extract_feature(image_fnames, extract_args)
        # feature_dict = {'dinov2': torch.zeros((len(image_fnames), 384, 100, 200))}
        if self.config.enable_cache:
            cache_dir.mkdir(exist_ok=True)
            
            cache_dict = {
                "args": extract_args.id_dict(),\
                "image_fnames": image_fnames,
                "feature_dict": feature_dict
            }
            
            torch.save(cache_dict, cache_path)
            CONSOLE.print(f"Saved DINO features to cache at {cache_path}")
        return feature_dict
    
    def __resize_tensor(self, input_tensor, target_shape):
        assert(input_tensor.ndim >= 2 and input_tensor.ndim <= 3)
        h, w = target_shape

        if input_tensor.ndim == 2:
            input_tensor = input_tensor.unsqueeze(0)
            
        if self.upsampler is None or self.upsampler.size != (h, w):
            self.upsampler = torch.nn.Upsample(size=(h, w), mode='nearest')
        img = self.upsampler(input_tensor.unsqueeze(0)).squeeze(0)
        
        if input_tensor.ndim == 2:
            img = img.squeeze(0)
        assert(img.shape[-2:] == target_shape)
        assert(img.ndim == input_tensor.ndim)
        return img
    
    def __upsample_features(self, source_dict, camera_idx, target_shape=None):
        feature_dict = {}
        for feature_name in source_dict:
            _tmp = source_dict[feature_name][camera_idx]
            feature_dict[feature_name] = self.__resize_tensor(_tmp, target_shape) if target_shape is not None else _tmp
        return feature_dict
    
    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        camera, data = super().next_train(step)
        camera_idx = camera.metadata['cam_idx']
        data["feature_dict"] = self.__upsample_features(self.train_feature_dict, camera_idx, None)
        segmentation = self.train_segment[camera_idx].to(dtype=torch.uint8)
        data["segmentation"] = segmentation
        return camera, data
    
    def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
        camera, data = super().next_eval(step)
        camera_idx = camera.metadata['cam_idx']
        data["feature_dict"] = self.__upsample_features(self.eval_feature_dict, camera_idx, None)
        segmentation = self.eval_segment[camera_idx].to(dtype=torch.uint8)
        data["segmentation"] = segmentation
        return camera, data
