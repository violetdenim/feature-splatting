import os
import gc
import numpy as np
import torch
from argparse import ArgumentParser
from PIL import Image
from tqdm import tqdm, trange
# import cv2
from typing import List#, Any, Dict, Generator, 

# import torch.nn as nn
import torchvision.transforms as T
# import torch.nn.functional as F
from PIL import Image
# import matplotlib.pyplot as plt


def norm(x, m=[0.485, 0.456, 0.406], s=[0.229, 0.224, 0.225]):
    y = torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = (x[:, i, :, :] - m[i]) / s[i]
    return y

def pytorch_gc():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()


def interpolate_to_patch_size(img_bchw, longest_edge, patch_size):
    # Interpolate the image so that H and W are multiples of the patch size
    _, _, H, W = img_bchw.shape
    
    if W > H:
        ratio = longest_edge / W
    else:
        ratio = longest_edge / H
    W = int(W * ratio)
    H = int(H * ratio)
    
    target_H = H // patch_size * patch_size
    target_W = W // patch_size * patch_size
    img_bchw = torch.nn.functional.interpolate(img_bchw, size=(target_H, target_W))
    return img_bchw, target_H, target_W
    

@torch.no_grad()
def batch_extract_feature(image_paths: List[str], args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_upsampled_dino = args.upsampled
    
    print("Loading DINOv2 model...")
    if use_upsampled_dino:
        dinov2 = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=True).to(device)    
        resolution = 784 #
        patch_size = 14
    else:
        dinov2 = torch.hub.load('facebookresearch/dinov2', args.dinov2_model_name).to(device)
        patch_size = dinov2.patch_size
        resolution = args.dino_resolution
    

    ret_dict = {'dinov2': []}
    for i in trange(len(image_paths)):
        image = Image.open(image_paths[i])
        image = torch.tensor(np.float32(image)/255.).permute([2, 0, 1]).unsqueeze(0).to(device=device)
        image, target_H, target_W = interpolate_to_patch_size(image, resolution, patch_size)
        if use_upsampled_dino:
            features = dinov2(norm(image))
            features = features.cpu().squeeze(0)
        else:
            with torch.no_grad():
                features = dinov2.forward_features((image - 0.5) / 0.5)["x_norm_patchtokens"][0]
                features = features.cpu()
                features = features.reshape((target_H // patch_size, target_W // patch_size, -1)).permute([2, 0, 1])
        
        ret_dict['dinov2'].append(features)
    
    print(ret_dict['dinov2'][0].shape)
    del dinov2
    pytorch_gc()
    
    for k in ret_dict.keys():
        ret_dict[k] = torch.stack(ret_dict[k], dim=0)  # BCHW

    return ret_dict

if __name__ == "__main__":
    parser = ArgumentParser("Compute reference features for feature splatting")
    parser.add_argument("--source_path", "-s", required=True, type=str)
    parser.add_argument("--dino_resolution", type=int, default=800, help="Longest edge for DINOv2 feature generation")
    parser.add_argument("--dinov2_model_name", type=str, default='dinov2_vits14')
    parser.add_argument("--upsampled", type=bool, default=True)
    args = parser.parse_args()

    image_paths = [os.path.join(args.source_path, fn) for fn in os.listdir(args.source_path)]

    ret_dict = batch_extract_feature(image_paths, args)
    
    for k, v in ret_dict.items():
        print(f"{k}: {v.shape}")
    torch.save(ret_dict[k], f"{k}.pt")
    