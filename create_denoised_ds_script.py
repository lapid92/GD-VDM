import os

import PIL.Image as pil
import torch
from torchvision import transforms as T

from constants import DENOISED_DEPTH_PATH, DEPTH_MODEL_PATH, DEPTH_PATH
from helpers import find_recursive
from networks import GaussianDiffusion, UNet


def create_denoised_ds():
    """
    This script generates a denoised depth dataset by applying a pre-trained 
    Video Diffusion Model (VDM) to each video in the original depth dataset.
    The process involves applying forward diffusion noise to the original depth 
    videos and then denoising them using the pre-trained denoising model of the VDM.
    The resulting dataset is saved as the "denoised" depth dataset.
    """
    depth_data_path = DEPTH_PATH
    denoised_depth_data_path = DENOISED_DEPTH_PATH
    depth_model_path = DEPTH_MODEL_PATH

    # Diffusion model values
    im_size = 64
    frames = 10
    max_num_frames = 30
    channels = 3
    dim_mults = (1, 2, 4, 8)
    diff_steps = 1000

    # timestep for the noised and denoised process
    num_timestep = 300

    # Random seed
    seed = 42
    # Video transformation pipeline
    video_transform = T.Compose([
        T.ToTensor(),
        T.Resize(im_size),
        T.CenterCrop(im_size),
    ])

    # Set device (CUDA if available, otherwise CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed(seed) 
    else:
        device = torch.device("cpu")
        torch.manual_seed(seed) 

    # UNet model initialization
    unet_model = UNet(
        im_size=im_size,
        channels=channels,
        dim_mults=dim_mults
    ).to(device)

    # VDM initialization
    diffusion_model = GaussianDiffusion(
        denoise_fn=unet_model,
        image_size=im_size,
        num_frames=frames,
        channels=channels,
        timesteps=diff_steps
    ).to(device)

    # Load pre-trained diffusion model state dict
    diffusion_state_dict = torch.load(str(depth_model_path))
    diffusion_model.load_state_dict(diffusion_state_dict['ema'])
    diffusion_model = diffusion_model.to(device)

    # Iterate over all the depth maps videos and apply the denoised process
    processed_dirs = []
    with torch.no_grad():
        for root, dirs, files in os.walk(depth_data_path):
            for dir in dirs:
                if not dir in processed_dirs:
                    print(dir)
                    imgs = find_recursive(os.path.join(depth_data_path, dir))
                    img_list = sorted([x for x in imgs])
                    frame_list = img_list

                    # video longer than max_num_frames
                    if len(frame_list) > max_num_frames:
                        output_folder = os.path.join(
                            denoised_depth_data_path, dir)
                        os.makedirs(output_folder, exist_ok=True)
                        num_of_frames = len(frame_list)
                        num_frames_list = [0]
                        for i in range(max_num_frames, num_of_frames+1, max_num_frames):
                            num_frames_list.append(i)
                        if num_of_frames % max_num_frames != 0:
                            num_frames_list.append(num_frames_list)
                        for frames_idx in range(len(num_frames_list) - 1):
                            part_frame_list = frame_list[num_frames_list[frames_idx]:num_frames_list[frames_idx+1]]
                            img_input_list = []
                            img_name_list = []
                            for img in part_frame_list:
                                img_name = img.split('/')[-1]
                                input_image = pil.open(img).convert('RGB')
                                img_input_list.append(
                                    video_transform(input_image))
                                img_name_list.append(img_name)
                            # check shape
                            img_input_tensor = torch.stack(
                                img_input_list, dim=1).unsqueeze(0).to(device)
                            denoised_video, noised_video = diffusion_model.noise_sample(img=img_input_tensor, noise_timesteps=num_timestep,
                                                                                        denoise_timestep=num_timestep)
                            denoised_list = torch.split(
                                denoised_video, split_size_or_sections=1, dim=2)
                            for denoised_img, denoised_name in zip(denoised_list, img_name_list):
                                denoised_img = denoised_img.squeeze(
                                    2).squeeze(0)
                                out_file = os.path.join(
                                    output_folder, denoised_name)
                                im = T.ToPILImage()(denoised_img)
                                im.save(out_file)
                        print("finish", dir)
                    else:
                        output_folder = os.path.join(
                            denoised_depth_data_path, dir)
                        os.makedirs(output_folder, exist_ok=True)
                        img_input_list = []
                        img_name_list = []
                        for img in frame_list:
                            img_name = img.split('/')[-1]
                            input_image = pil.open(img).convert('RGB')
                            img_input_list.append(video_transform(input_image))
                            img_name_list.append(img_name)
                        # check shape
                        img_input_tensor = torch.stack(
                            img_input_list, dim=1).unsqueeze(0).to(device)
                        denoised_video, noised_video = diffusion_model.noise_sample(img=img_input_tensor, noise_timesteps=num_timestep,
                                                                                    denoise_timestep=num_timestep)
                        denoised_list = torch.split(
                            denoised_video, split_size_or_sections=1, dim=2)
                        for denoised_img, denoised_name in zip(denoised_list, img_name_list):
                            denoised_img = denoised_img.squeeze(2).squeeze(0)
                            out_file = os.path.join(
                                output_folder, denoised_name)
                            im = T.ToPILImage()(denoised_img)
                            im.save(out_file)
                        print("finish", dir)
                else:
                    print("Already done", dir)
        print('Denoised process done!')
