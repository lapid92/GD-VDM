# GD-VDM

This repository contains the code for training and running the Generated Depth - Video Diffusion Model (GD-VDM). The GD-VDM consists of several steps, which are explained below.

## Prerequisites
Before running the script, ensure that you modify the necessary parameters and paths to
fit your requirements.

## Steps

The main script serves as the entry point for training the GD-VDM. It encompasses the entire training procedure, which includes the following steps:

1. Training the Video Diffusion Model (VDM) on depth videos.
2. Creating a "denoised" depth video dataset:
- Apply forward diffusion noise to the original depth videos.
- Denoise the videos using the pre-trained denoising model of the VDM.
3. Training the Video to Video Diffusion Model (Vid2Vid-DM) on a paired photorealistic-
"denoised" depth video dataset:
- The Vid2Vid-DM learns the mapping from "denoised" depth videos to photorealistic real-world videos.
- Load the weights of the VDM, trained on depth map videos, into the Depth U-Net of the Vid2Vid model, but do not freeze them.
4. Generating samples using the GD-VDM framework:
- Synthesize depth videos using the VDM trained on depth videos.
- Pass the generated depth videos to the Vid2Vid-DM to create photorealistic real-world videos.

Note: You can also run the individual scripts for each step in the procedure if you want to perform specific tasks separately.

Enjoy using GD-VDM!
