import copy
from pathlib import Path
from typing import Optional, Union

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.utils.data import DataLoader

from helpers import video_tensor_to_gif
from networks import EMA

# The Trainer class is responsible for training a video diffusion model.
# The class supports both conditional and unconditional diffusion models.
# Samples are generated and saved as GIF files for visualization.


class Trainer(object):
    def __init__(
        self,
        diffusion_model: nn.Module,
        results_folder: str,
        device: torch.device,
        batch_size: int,
        lr: float,
        save_checkpoint_every: int,
        save_sample_every: int,
        train_num_steps: int,
        train_dl: DataLoader,
        val_dl: Optional[DataLoader] = None,
        validate_every: Optional[int] = None,
        ema_decay: float = 0.995,
        update_ema_every: int = 10,
        step_start_ema: int = 2000,
        warmup_iters: int = 0,
        warmup_factor: Optional[float] = None,
        seed: int=42
    ):
        """
    Trainer class for training a video diffusion model.

    Args:
        diffusion_model (nn.Module): The video diffusion model to be trained.
        results_folder (str): Path to the folder where training results will be saved.
        device (str): Device to be used for training (e.g., 'cpu', 'cuda').
        batch_size (int): Batch size for training.
        lr (float): Learning rate for the optimizer.
        save_checkpoint_every (int): Number of steps between saving model checkpoints.
        save_sample_every (int): Number of steps between sampling videos.
        train_num_steps (int): Total number of training steps.
        train_dl (object): Training data loader.
        val_dl (object, optional): Validation data loader, relevant only for conditional models. Defaults to None.
        validate_every (int, optional): Number of steps between validation evaluations, relevant only for conditional models. Defaults to None.
        ema_decay (float, optional): Decay rate for exponential moving average. Defaults to 0.995.
        update_ema_every (int, optional): Number of steps between updating the EMA model. Defaults to 10.
        step_start_ema (int, optional): Step to start updating the EMA model. Defaults to 2000.
        warmup_iters (int, optional): Number of warmup iterations for learning rate scheduling. Defaults to 0.
        warmup_factor (float, optional): Start factor for learning rate scheduling. Defaults to None.
    """

        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.device = device

        self.step_start_ema = step_start_ema
        self.save_sample_every = save_sample_every
        self.save_checkpoint_every = save_checkpoint_every

        self.batch_size = batch_size
        self.image_size = diffusion_model.image_size
        self.train_num_steps = train_num_steps
        self.channels = diffusion_model.channels
        self.num_frames = diffusion_model.num_frames

        # Check if the diffusion model has a conditional architecture
        if hasattr(diffusion_model, 'cond_fn'):
            self.cond_arch = True
        else:
            self.cond_arch = False

        self.dl = train_dl

        # Set validation dataloader
        if val_dl is None:
            self.val_dl = train_dl
        else:
            self.val_dl = val_dl

        # Set validation frequency
        if validate_every is None:
            self.validate_every = self.save_sample_every
        else:
            self.validate_every = validate_every

        self.opt = Adam(diffusion_model.parameters(), lr=lr)

        # Set up learning rate scheduling
        if warmup_iters > 0:
            self.scheduling = True
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.opt, start_factor=warmup_factor, total_iters=warmup_iters)
        else:
            self.scheduling = False
            self.scheduler = self.opt

        self.step = 0

        self.scaler = GradScaler(enabled=True)
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True, parents=True)

        self.samples_folder = Path(results_folder + '/samples')
        self.samples_folder.mkdir(exist_ok=True, parents=True)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset the parameters of the EMA model to match the current model.
        """
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        """
        Perform a step in updating the Exponential Moving Average (EMA) model.

        If the current step is below the specified starting step for EMA updates,
        the EMA model parameters are reset to match the current model parameters.
        Otherwise, the EMA model is updated by averaging its parameters with the
        parameters of the current model.
        """
        if self.step < self.step_start_ema:
            # Reset EMA model parameters to match current model parameters
            self.reset_parameters()
            return
        # Update EMA model by averaging its parameters with the current model parameters
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self,
             milestone: str,
             path: Optional[str] = None):
        """
        Save the current training state, including the step, model parameters, EMA model parameters,
        and scaler state.

        Args:
            milestone (str): Milestone identifier for the saved model.
            path (str, optional): Path to save the model. If not provided, the model is saved in the
                results folder with a filename based on the milestone. Defaults to None.
        """
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        if path is None:
            # Save the data to a file in the results folder with a filename based on the milestone
            torch.save(data, str(self.results_folder /
                       f'model-{milestone}.pt'))
        else:
            # Save the data to the specified path
            torch.save(data, path)

    def load(self,
             milestone: Union[int, str],
             **kwargs):
        """
        Load a specific training state from a checkpoint file.

        Args:
            milestone (int or str): The milestone identifier or -1 to load the latest checkpoint.
        """
        if milestone == -1:
            # If milestone is -1, find the latest checkpoint by parsing the milestones from the filenames
            all_milestones = [int(p.stem.split('-')[-1])
                              for p in Path(self.results_folder).glob('**/*.pt')]
            assert len(
                all_milestones) > 0, 'need to have at least one milestone to load the model from latest checkpoint'
            milestone = max(all_milestones)

        # Load the checkpoint data from the file
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))

        # Update the current step with the loaded step
        self.step = data['step']

        # Load the model's state dict from the checkpoint data
        self.model.load_state_dict(data['model'], **kwargs)

        # Load the EMA model's state dict from the checkpoint data
        self.ema_model.load_state_dict(data['ema'], **kwargs)

        # Load the scaler's state dict from the checkpoint data
        self.scaler.load_state_dict(data['scaler'])

    def train(
        self,
        val_cond_scale: float = 0.
    ):
        """
        Train the video diffusion model.

        Args:
            val_cond_scale (float, optional): Scale factor for conditioning during validation. Defaults to 0.
        """
        while self.step < self.train_num_steps:
            # Fetch the next batch of videos
            image = next(self.dl)
            if self.cond_arch:
                # Conditional architecture: extract real videos and conditional videos
                real_image = image['image'].to(self.device)
                cond_img = image['cond'].to(self.device)

                with autocast(enabled=True):
                    # Compute loss and perform backward pass
                    loss = self.model(
                        real_image,
                        cond_img,
                    )
                    self.scaler.scale(loss).backward()
            else:
                # Non-conditional architecture
                image = image['image'].to(self.device)

                with autocast(enabled=True):
                    # Compute loss and perform backward pass
                    loss = self.model(image)
                    self.scaler.scale(loss).backward()

            print(f'{self.step}: {loss.item()}')

            # Update model parameters
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()

            if self.scheduling:
                self.scheduler.step()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            # Saving model
            if self.step != 0 and self.step % self.save_checkpoint_every == 0:
                milestone = self.step // self.save_checkpoint_every
                self.save(milestone)

            # Sampling videos
            if self.step != 0 and (self.step % self.save_sample_every == 0):
                milestone = self.step // self.save_sample_every
                if self.cond_arch:
                    image = next(self.val_dl)
                    real_image = image['image'].to(self.device)
                    cond_img = image['cond'].to(self.device)

                    with torch.no_grad():
                        pred_vid = self.ema_model.sample(
                            cond_img=cond_img, cond_scale=val_cond_scale)
                        for vid_num in range(self.batch_size):
                            video_path = str(
                                self.samples_folder / str(f'{milestone}_{vid_num}.gif'))
                            video_tensor_to_gif(
                                pred_vid[vid_num, :, :, :, :], video_path)
                else:
                    with torch.no_grad():
                        pred_vid = self.ema_model.sample(
                            batch_size=self.batch_size)
                    for vid_num in range(self.batch_size):
                        video_path = str(self.samples_folder /
                                         str(f'{milestone}_{vid_num}.gif'))
                        video_tensor_to_gif(
                            pred_vid[vid_num, :, :, :, :], video_path)

            # Validation
            if self.cond_arch and self.step % self.validate_every == 0:
                image = next(self.val_dl)
                real_image = image['image'].to(self.device)
                cond_img = image['cond'].to(self.device)
                with torch.no_grad():
                    self.ema_model.eval()
                    val_loss = self.ema_model(
                        real_image,
                        cond_img
                    )
                    self.ema_model.train()
                print(f'val loss {self.step}: {val_loss.item()}')

            self.step += 1
        print('training completed')
