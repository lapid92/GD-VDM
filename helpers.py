# This file contains the helper function used in the project's scripts.

import os
from typing import List

import torch
import torchvision.transforms as T
from PIL import Image


def video_tensor_to_gif(tensor: torch.Tensor,
                        path: str) -> List[Image.Image]:
    """
    Converts a video tensor to a GIF file.

    Args:
        tensor (torch.Tensor): Video tensor.
        path (str): Path to save the GIF file.

    Returns:
        List[Image.Image]: List of PIL Image objects representing each frame of the video.

    """
    images = map(T.ToPILImage(), tensor.unbind(dim=1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all=True, append_images=rest_imgs)
    return images


def find_recursive(root_dir: str, ext: str = '.png') -> List[str]:
    """
    Recursively finds files with a specific extension in a directory.

    Args:
        root_dir (str): Root directory to search in.
        ext (str): File extension to filter by (default: '.png').

    Returns:
        List[str]: List of file paths matching the given extension.

    """
    files = []
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(ext):
                files.append(os.path.join(root, filename))
    return files
