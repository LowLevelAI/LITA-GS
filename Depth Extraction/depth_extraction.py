import os
from argparse import ArgumentParser
from marigold import MarigoldPipeline
import logging
import numpy as np
import torch
from tqdm.auto import tqdm
from glob import glob
from PIL import Image
import cv2

if __name__ == '__main__':
    
    parser = ArgumentParser(description="Monocular depth/normal maps parameters")
    parser.add_argument('-s', type=str, default= 'data/LOM_low_bike_colmap', help='Path to the data directory')
    args = parser.parse_args()
    input_dir =  args.s + "/images_low"
    output_dir_color = args.s + "/depth_maps"


    # monocular depth maps
    ckpt_path = 'Depth Extraction/checkpoint/marigold-depth-lcm-v1-0'
    pipe = MarigoldPipeline.from_pretrained(ckpt_path)
    pipe = pipe.to("cuda")
    EXTENSION_LIST = [".jpg", ".jpeg", ".png"]

    # Image list
    rgb_filename_list = glob(os.path.join(input_dir, "*"))
    rgb_filename_list = [
        f for f in rgb_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST
    ]
    rgb_filename_list = sorted(rgb_filename_list)

    # Create output folders
    os.makedirs(output_dir_color, exist_ok=True)

    # Run Inference
    with torch.no_grad():
        for rgb_path in tqdm(rgb_filename_list, desc=f"Estimating depth", leave=True):
            # Read input image
            input_image = Image.open(rgb_path)

            # Predict depth
            pipeline_output = pipe(
                input_image,
            )

            depth_pred: np.ndarray = pipeline_output.depth_np
            depth_colored: Image.Image = pipeline_output.depth_colored

            # Save as npy
            rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
            pred_name_base = rgb_name_base


            # Colorize
            colored_save_path = os.path.join(
                output_dir_color, f"depth_{pred_name_base}.png"
            )
            if os.path.exists(colored_save_path):
                logging.warning(f"Existing file: '{colored_save_path}' will be overwritten")
            cv2.imwrite(colored_save_path, depth_pred*255)

