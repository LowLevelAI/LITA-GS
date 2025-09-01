import cv2
import einops
import numpy as np
import torch
import random
import glob
import os
import argparse


from ciconv2d0 import CIConv2d
from PIL import Image


def rgb(t): return (
        np.clip((t[0] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose([1, 2, 0]), 0, 1) * 255).astype(
    np.uint8)

def gray(t): return (
        np.clip((t[0][0] if len(t.shape) == 4 else t[0]).detach().cpu().numpy(), 0, 1) * 255).astype(
    np.uint8)

if __name__ == '__main__':

    inp_root = 'data/LOM_low_bike_colmap'
    input_folder = inp_root + '/images_low'
    output_folder = inp_root + '/W_0.8'
    
    extraction_model = CIConv2d('W', k=3, scale=0.8)
    extraction_model = extraction_model.cuda()


    # extract prior
    img_list = glob.glob(f"{input_folder}/*.*")
    print(f"Find {len(img_list)} files in {input_folder}")

    H_folder = output_folder
    os.makedirs(H_folder, exist_ok=True)

    for img_path in img_list:
        
        input_image = cv2.imread(img_path)
        
        input_tensor = (torch.from_numpy(input_image.copy()).cuda().to(dtype=torch.float32) / 255.0).unsqueeze(0)
        
        input_tensor = einops.rearrange(input_tensor, 'b h w c -> b c h w').clone()

        with torch.no_grad():
            features = extraction_model(input_tensor)
        H_path = os.path.join(H_folder, os.path.basename(img_path))

        cv2.imwrite(H_path, gray(features))
        

    
