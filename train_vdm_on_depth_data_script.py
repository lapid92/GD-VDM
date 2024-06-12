import os
import re

import torch

from constants import DEPTH_MODEL_PATH, DEPTH_OUTDIR, DEPTH_PATH
from datasets import InfiniteSampler, VideoDataset
from helpers import video_tensor_to_gif
from networks import GaussianDiffusion, UNet
from trainer import Trainer


def train_vdm_on_depth_videos():
    """
    The purpose of this script is to train the Video Diffusion Model (VDM) on depth maps videos.
    """

    # Set the paths for data and model weights
    depth_data_path = DEPTH_PATH
    output_model_path = DEPTH_MODEL_PATH

    # Set configuration parameters
    im_size = 64
    frames = 10
    outdir = DEPTH_OUTDIR
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
    train_num_steps = 1300000
    channels = 3
    dim_mults = (1, 2, 4, 8)
    desc = 'depth_cityscapes'
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

    # Initialize the dataset
    depth_ds = VideoDataset(ds_path=depth_data_path,
                            num_frames=frames,
                            image_size=im_size,
                            hflip=hflip)
    depth_sampler = InfiniteSampler(
        dataset=depth_ds, shuffle=shuffle, seed=seed)
    depth_dl = iter(torch.utils.data.DataLoader(dataset=depth_ds, sampler=depth_sampler, batch_size=batch_size,
                                                num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last))

    # Initialize the U-Net
    model = UNet(
        im_size=im_size,
        channels=channels,
        dim_mults=dim_mults).to(device)

    # Initialize the diffusion model
    diffusion = GaussianDiffusion(
        denoise_fn=model,
        image_size=im_size,
        num_frames=frames,
        channels=channels,
        timesteps=diff_steps
    ).to(device)

    # Initialize the trainer
    trainer = Trainer(
        diffusion_model=diffusion,
        results_folder=run_dir,
        device=device,
        batch_size=batch_size,
        lr=lr,
        save_checkpoint_every=save_checkpoint_every,
        save_sample_every=save_sample_every,
        train_num_steps=train_num_steps,
        train_dl=depth_dl,
    )

    # Start the training loop
    trainer.train()

    # Save the trained model
    trainer.save(milestone='latest', path=output_model_path)

    # Construct the output sample dir
    samples_dir = os.path.join(run_dir, 'latest_samples')
    os.makedirs(samples_dir, exist_ok=True)

    # Generate depth videos from the diffusion model
    pred_vid = trainer.ema_model.sample(batch_size=batch_size)

    # Save the generated videos as GIFs
    for vid_num in range(batch_size):
        video_path = os.path.join(
            samples_dir, str(f'latest_sample_{vid_num}.gif'))
        video_tensor_to_gif(pred_vid[vid_num, :, :, :, :], video_path)
