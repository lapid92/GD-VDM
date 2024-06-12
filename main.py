# Main Training Script: Generated Depth - Video Diffusion Model (GD-VDM)

"""
This script is the main entry point for training the Generated Depth - Video Diffusion Model (GD-VDM).
It encompasses the entire training procedure, consisting of the following steps:

1. Training Video Diffusion Model (VDM) on depth videos.
   - The VDM is trained to generate denoised depth videos.

2. Creating a "denoised" depth video dataset.
   - This step involves applying forward diffusion noise to the original depth videos and 
    then denoising them using the pre-trained denoising model of the VDM.

3. Training Video to Video Diffusion Model (Vid2Vid-DM) on a paired photorealistic-denoised_depth video dataset.
   - The Vid2Vid-DM is trained to learn the mapping from denoised depth videos to photorealistic real-world videos.
   - The weights of the VDM, trained on depth maps videos, are loaded into the Depth UNet of the Vid2Vid model,
    but they are not frozen.

4. Generating samples using the GD-VDM framework.
   - Synthesizing depth videos using the VDM trained on depth videos.
   - Passing the generated depth videos to the Vid2Vid-DM to create photorealistic real-world videos.

Please make sure to modify the necessary parameters and paths before running this script.
"""

from create_denoised_ds_script import create_denoised_ds
from generate_gd_vdm_script import generate_gd_vdm_samples
from train_gd_vdm_script import train_gd_vdm
from train_vdm_on_depth_data_script import train_vdm_on_depth_videos

# train VDM on depth data
train_vdm_on_depth_videos()

# create denoised depth ds
create_denoised_ds()

# train GD-VDM
train_gd_vdm()

# generate video with GD-VDM
generate_gd_vdm_samples()
