import gc, os, cv2
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type
from nerfstudio.cameras.cameras import Cameras, CameraType

import numpy as np
import torch
from jaxtyping import Float
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanager,
    FullImageDatamanagerConfig,
)
from nerfstudio.utils.rich_utils import CONSOLE

from feature_splatting.feature_extractor_cfg import SAMCLIPArgs

# SAMCLIP
from feature_splatting.feature_extractor import batch_extract_feature

feat_type_to_extract_fn = {
    "CLIP": None,
    "DINO": None,
    "SAMCLIP": batch_extract_feature,
}

feat_type_to_args = {
    "CLIP": None,
    "DINO": None,
    "SAMCLIP": SAMCLIPArgs,
}

feat_type_to_main_feature_name = {
    "CLIP": "clip",
    "DINO": "dino",
    "SAMCLIP": "samclip",
}

@dataclass
class FeatureSplattingDataManagerConfig(FullImageDatamanagerConfig):
    _target: Type = field(default_factory=lambda: FeatureSplattingDataManager)
    feature_type: Literal["CLIP", "DINO", "SAMCLIP"] = "SAMCLIP"
    """Feature type to extract."""
    enable_cache: bool = True
    """Whether to cache extracted features."""

class FeatureSplattingDataManager(FullImageDatamanager):
    config: FeatureSplattingDataManagerConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Extract features
        self.feature_dict = self.extract_features()

        # Split into train and eval features
        self.train_feature_dict = {}
        self.eval_feature_dict = {}
        feature_dim_dict = {}
        for feature_name in self.feature_dict:
            assert len(self.feature_dict[feature_name]) == len(self.train_dataset) + len(self.eval_dataset)
            self.train_feature_dict[feature_name] = self.feature_dict[feature_name][: len(self.train_dataset)]
            self.eval_feature_dict[feature_name] = self.feature_dict[feature_name][len(self.train_dataset) :]
            feature_dim_dict[feature_name] = self.feature_dict[feature_name].shape[1:]  # c, h, w
        assert len(self.eval_feature_dict[feature_name]) == len(self.eval_dataset)

        del self.feature_dict

        # Set metadata, so we can initialize model with feature dimensionality
        self.train_dataset.metadata["feature_type"] = self.config.feature_type
        self.train_dataset.metadata["feature_dim_dict"] = feature_dim_dict
        self.train_dataset.metadata["main_feature_name"] = feat_type_to_main_feature_name[self.config.feature_type]
        self.train_dataset.metadata["clip_model_name"] = feat_type_to_args[self.config.feature_type].clip_model_name


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
                
        # Garbage collect
        torch.cuda.empty_cache()
        gc.collect()
    
    def extract_features(self) -> Dict[str, Float[torch.Tensor, "n h w c"]]:
        # Extract features
        if self.config.feature_type not in feat_type_to_extract_fn:
            raise ValueError(f"Unknown feature type {self.config.feature_type}")
        extract_fn = feat_type_to_extract_fn[self.config.feature_type]
        extract_args = feat_type_to_args[self.config.feature_type]
        image_fnames = self.train_dataset.image_filenames + self.eval_dataset.image_filenames
        # For dev purpose, visually tested image_fnames order matches camera_idx. NS seems to internally sort valid image_fnames.
        # self.feature_image_fnames = image_fnames

        # If cache exists, load it and validate it. We save it to the dataset directory.
        cache_dir = self.config.dataparser.data
        cache_path = cache_dir / f"feature_splatting_{self.config.feature_type.lower()}_features.pt"
        if self.config.enable_cache and cache_path.exists():
            cache_dict = torch.load(cache_path)
            if cache_dict.get("image_fnames") != image_fnames:
                CONSOLE.print("Image filenames have changed, cache invalidated...")
            elif cache_dict.get("args") != extract_args.id_dict():
                CONSOLE.print("Feature extraction args have changed, cache invalidated...")
            else:
                return cache_dict["feature_dict"]

        # Cache is invalid or doesn't exist, so extract features
        CONSOLE.print(f"Extracting {self.config.feature_type} features for {len(image_fnames)} images...")
        feature_dict = extract_fn(image_fnames, extract_args)
        if self.config.enable_cache:
            cache_dict = {"args": extract_args.id_dict(), "image_fnames": image_fnames, "feature_dict": feature_dict}
            cache_dir.mkdir(exist_ok=True)
            torch.save(cache_dict, cache_path)
            CONSOLE.print(f"Saved {self.config.feature_type} features to cache at {cache_path}")
        return feature_dict

    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        camera, data = super().next_train(step)
        camera_idx = camera.metadata['cam_idx']
        feature_dict = {}
        for feature_name in self.train_feature_dict:
            feature_dict[feature_name] = self.train_feature_dict[feature_name][camera_idx]
        data["feature_dict"] = feature_dict
        segmentation = self.train_segment[camera_idx].to(dtype=torch.uint8)
        data["segmentation"] = segmentation
        return camera, data
    
    def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
        camera, data = super().next_eval(step)
        camera_idx = camera.metadata['cam_idx']
        feature_dict = {}
        for feature_name in self.eval_feature_dict:
            feature_dict[feature_name] = self.eval_feature_dict[feature_name][camera_idx]
        data["feature_dict"] = feature_dict
        segmentation = self.eval_segment[camera_idx].to(dtype=torch.uint8)
        data["segmentation"] = segmentation
        return camera, data
