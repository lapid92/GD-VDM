import os
import re

import torch

from constants import (DENOISED_DEPTH_PATH, DEPTH_MODEL_PATH,
                       GD_VDM_MODEL_PATH, REAL_OUTDIR, REAL_PATH, ROOT_PATH)
from datasets import InfiniteSampler, PairedVideoDataset
from helpers import video_tensor_to_gif
from networks import (Cond_UNet, Extented_GaussianDiffusionModel,
                      GaussianDiffusion, UNet)
from trainer import Trainer


def train_gd_vdm():
    """
    The purpose of this script is to train the Video to Video Diffusion Model (Vid2Vid-DM).
    In the training process, we utilize the weights of the Depth U-Net that were obtained from a previously trained
    VDM on depth videos.
    However, instead of freezing the weights, we allow them to be updated during training.
    The model is trained using paired videos, which consist of real-world videos alongside their corresponding "denoised" depth videos.
    """

    # Set the paths for data and model weights
    root_data_path = ROOT_PATH
    real_data_path = REAL_PATH
    depth_data_path = DENOISED_DEPTH_PATH

    depth_model_path = DEPTH_MODEL_PATH
    output_model_path = GD_VDM_MODEL_PATH

    # Set configuration parameters
    im_size = 64
    frames = 10
    outdir = REAL_OUTDIR
    diff_steps = 1000
    batch_size = 9
    num_workers = 9
    shuffle = True
    drop_last = True
    pin_memory = True
    hflip = True
    lr = 1e-4
    save_checkpoint_every = 50000
    save_sample_every = 50000
    validate_every = 50000
    train_num_steps = 1300000
    warmup_iters = 300
    warmup_factor = 0.05
    channels = 3
    cond_dropout = 0.2
    cond_scale = 1.4
    dim_mults = (1, 2, 4, 8)
    desc = 'gd_vdm_cityscapes'
    load_pretrained_depth_unet_weights = True
    seed = 42

    # Check if CUDA is available and set the device accordingly
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

    # Initialize the dataset
    cond_ds = PairedVideoDataset(root_path=root_data_path, real_path=real_data_path, cond_path=depth_data_path,
                                 num_frames=frames,
                                 image_size=im_size,
                                 hflip=hflip)
    cond_sampler = InfiniteSampler(dataset=cond_ds, shuffle=shuffle, seed=seed)
    cond_dl = iter(torch.utils.data.DataLoader(dataset=cond_ds, sampler=cond_sampler, batch_size=batch_size,
                                               num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last))

    # Use the same validation dataset as the training dataset
    val_dl = cond_dl

    # Initialize the Depth U-Net and Video U-Net models
    Depth_UNet = UNet(
        im_size=im_size,
        channels=channels,
        dim_mults=dim_mults,
        out_list=True).to(device)

    Video_UNet = Cond_UNet(
        im_size=im_size,
        channels=channels,
        dim_mults=dim_mults).to(device)

    # Initialize the diffusion model
    diffusion = Extented_GaussianDiffusionModel(
        denoise_fn=Video_UNet,
        cond_fn=Depth_UNet,
        image_size=im_size,
        num_frames=frames,
        channels=channels,
        cond_channels=channels,
        timesteps=diff_steps,
        cond_dropout=cond_dropout
    ).to(device)

    # Load pretrained weights of the Depth U-Net if specified
    if load_pretrained_depth_unet_weights:
        print("Resuming model from ", depth_model_path)
        data = torch.load(str(depth_model_path))['ema']
        for key in list(data):
            if key.startswith('denoise_fn'):
                data[key.replace('denoise_fn.', '')] = data.pop(key)
            else:
                data.pop(key)
        diffusion.cond_fn.load_state_dict(data)
        diffusion = diffusion.cuda()

    # Initialize the trainer
    trainer = Trainer(
        diffusion_model=diffusion,
        results_folder=run_dir,
        device=device,
        batch_size=batch_size,
        lr=lr,
        save_checkpoint_every=save_checkpoint_every,
        save_sample_every=save_sample_every,
        validate_every=validate_every,
        train_num_steps=train_num_steps,
        train_dl=cond_dl,
        val_dl=val_dl,
        warmup_iters=warmup_iters,
        warmup_factor=warmup_factor
    )

    # Start the training loop
    trainer.train(val_cond_scale=cond_scale)

    # Save the trained model
    trainer.save(milestone='latest', path=output_model_path)

    # Construct the output sample dir
    samples_dir = os.path.join(run_dir, 'latest_samples')
    os.makedirs(samples_dir, exist_ok=True)

    # Initialize the U-Net of the Depth VDM model
    Depth_VDM = UNet(
        im_size=im_size,
        channels=channels,
        dim_mults=dim_mults).to(device)

    # Initialize the depth diffusion model
    depth_diffusion = GaussianDiffusion(
        denoise_fn=Depth_VDM,
        image_size=im_size,
        num_frames=frames,
        channels=channels,
        timesteps=diff_steps
    ).to(device)

    # Load pretrained weights for the depth diffusion model
    print("Resuming model from ", depth_model_path)
    data = torch.load(str(depth_model_path))
    depth_diffusion.load_state_dict(data['ema'])
    depth_diffusion = depth_diffusion.to(device)

    # Generate depth videos from the depth diffusion model
    depth_vid = depth_diffusion.sample(batch_size=batch_size)

    # Generate real-world videos from the trained diffusion model using the generated depth videos as conditional input
    real_vid = trainer.ema_model.sample(
        cond_img=depth_vid, cond_scale=cond_scale)

    # Save the generated videos as GIFs
    for vid_num in range(batch_size):
        video_path = os.path.join(
            samples_dir, str(f'latest_sample_{vid_num}.gif'))
        video_tensor_to_gif(real_vid[vid_num, :, :, :, :], video_path)
