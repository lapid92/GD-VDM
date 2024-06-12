import os
import re

import torch

from constants import DEPTH_MODEL_PATH, GD_VDM_MODEL_PATH, SAMPLES_OUTDIR
from helpers import video_tensor_to_gif
from networks import (Cond_UNet, Extented_GaussianDiffusionModel,
                      GaussianDiffusion, UNet)


def generate_gd_vdm_samples():
    """
    This script generates samples of real-world videos using the Generated Depth - Video Diffusion Model (GD-VDM) framework.
    The process involves the following steps:
    1. Generating depth videos using a Video Diffusion Model (VDM) that has been trained on depth videos.
    2. Passing these depth videos to the Video to Video Diffusion Model (Vid2Vid-DM) to generate real-world videos.
    """
    depth_model_path = DEPTH_MODEL_PATH
    output_model_path = GD_VDM_MODEL_PATH
    outdir = SAMPLES_OUTDIR

    # Diffusion model values
    im_size = 64
    frames = 10
    diff_steps = 1000
    batch_size = 9
    channels = 3
    cond_dropout = 0.2
    cond_scale = 1.4
    dim_mults = (1, 2, 4, 8)
    desc = 'gd_vdm_cityscapes'

    # Random seed
    seed = 42

    # Set device (CUDA if available, otherwise CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed(seed) 
    else:
        device = torch.device("cpu")
        torch.manual_seed(seed) 

    # Pick output directory
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(
            outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    assert not os.path.exists(run_dir)
    print('Creating output directory...')
    print(run_dir)
    os.makedirs(run_dir)

    # Depth UNet
    Depth_VDM_UNet = UNet(
        im_size=im_size,
        channels=channels,
        dim_mults=dim_mults).to(device)

    # Depth VDM
    depth_vdm = GaussianDiffusion(
        denoise_fn=Depth_VDM_UNet,
        image_size=im_size,
        num_frames=frames,
        channels=channels,
        timesteps=diff_steps
    ).to(device)

    # Resume the Depth VDM weights
    print("Resuming model from ", depth_model_path)
    data = torch.load(str(depth_model_path))
    depth_vdm.load_state_dict(data['ema'])
    depth_vdm = depth_vdm.to(device)

    # GD-VDM Depth UNet
    Depth_UNet = UNet(
        im_size=im_size,
        channels=channels,
        dim_mults=dim_mults,
        out_list=True).to(device)

    # GD-VDM Video UNet
    Video_UNet = Cond_UNet(
        im_size=im_size,
        channels=channels,
        dim_mults=dim_mults).to(device)

    # GD-VDM
    gd_vdm = Extented_GaussianDiffusionModel(
        denoise_fn=Video_UNet,
        cond_fn=Depth_UNet,
        image_size=im_size,
        num_frames=frames,
        channels=channels,
        cond_channels=channels,
        timesteps=diff_steps,
        cond_dropout=cond_dropout
    ).to(device)

    # Resume the GD-VDM weights
    print("Resuming GD-VDM from ", output_model_path)
    data = torch.load(str(output_model_path))
    gd_vdm.load_state_dict(data['ema'])
    gd_vdm = gd_vdm.to(device)

    # Sample depth videos
    depth_vid = depth_vdm.sample(batch_size=batch_size)

    # Sample real-world videos using the generated depth videos
    real_vid = gd_vdm.sample(
        cond_img=depth_vid, cond_scale=cond_scale)

    # Save the real-world samples
    for vid_num in range(batch_size):
        video_path = os.path.join(run_dir, str(f'latest_sample_{vid_num}.gif'))
        video_tensor_to_gif(real_vid[vid_num, :, :, :, :], video_path)
