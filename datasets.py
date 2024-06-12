# This file includes the dataloaders and sampler utilized in the project's scripts.
# If you intend to use these classes, please ensure that your data is organized
# in the following manner:
#
# cityscapes
# └───images
#     └───seq0001
#         └───000001.png
#         └───000002.png
#         ...
#     └───seq0002
#         └───000001.png
#         └───000002.png
#         ...
#     ...
# └───depth
#     └───seq0001
#         └───000001.png
#         └───000002.png
#         ...
#     └───seq0002
#         └───000001.png
#         └───000002.png
#         ...
#     ...

import os
import random
from typing import IO, Callable, Iterator, List, Tuple

import numpy as np
import PIL.Image
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms as T


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self,
                 ds_path: str,
                 num_frames: int,
                 max_num_frames: int = 1000000,
                 image_size: int = 64,
                 hflip: bool = False):
        """
        Dataset class for video.

        Args:
            ds_path (str): Root path of the dataset.
            num_frames (int): Number of frames to sample from video.
            max_num_frames (int, optional): Maximum number of frames to consider from video. Defaults to 1000000.
            image_size (int, optional): Resolution of the output images. Defaults to 64.
            hflip (bool, optional): Whether to apply horizontal flipping augmentation. Defaults to False.
        """
        self.path = ds_path
        self.num_frames = num_frames
        self.max_num_frames = max_num_frames
        self.image_size = image_size

        self.hflip = hflip
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize(self.image_size),
            T.CenterCrop(self.image_size),
        ])

        def listdir_full_paths(d: str) -> list[str]:
            """
            Returns a sorted list of full paths to files in the given directory.

            Args:
                d (str): Directory path.

            Returns:
                list[str]: Sorted list of full file paths.
            """
            return sorted([os.path.join(d, x) for x in os.listdir(d)])

        if os.path.isdir(self.path):
            # We assume that the depth is 2
            self._all_objects = {o for d in listdir_full_paths(self.path) for o in (
                ([d] + listdir_full_paths(d)) if os.path.isdir(d) else [d])}
            self._all_objects = {os.path.relpath(o, start=os.path.dirname(
                self.path)) for o in {self.path}.union(self._all_objects)}
        else:
            raise IOError('Path must be a directory')

        PIL.Image.init()
        self._video_dir2frames = {}
        objects = sorted([d for d in self._all_objects])
        root_path_depth = len(os.path.normpath(
            objects[0]).split(os.path.sep))
        curr_d = objects[1]  # Root path is the first element

        for o in objects[1:]:
            curr_obj_depth = len(os.path.normpath(o).split(os.path.sep))
            # change PIL.Image.EXTENSION to user choice?
            if self._file_ext(o) in PIL.Image.EXTENSION:
                assert o.startswith(
                    curr_d), f"Object {o} is out of sync. It should lie inside {curr_d}"
                assert curr_obj_depth == root_path_depth + \
                    2, "Frame images should be inside directories"
                if not curr_d in self._video_dir2frames:
                    self._video_dir2frames[curr_d] = []
                self._video_dir2frames[curr_d].append(o)
            else:
                # We encountered a new directory
                assert curr_obj_depth == root_path_depth + \
                    1, f"Video directories should be inside the root dir. {o} is not."
                if curr_d in self._video_dir2frames:
                    sorted_files = sorted(self._video_dir2frames[curr_d])
                    self._video_dir2frames[curr_d] = sorted_files
                curr_d = o
        self._video_idx2frames = [
            frames for frames in self._video_dir2frames.values()]

        if len(self._video_idx2frames) == 0:
            raise IOError('No videos found in the specified archive')

        dummy_vid = self._load_raw_frames(
            self._video_idx2frames, 0, np.arange(num_frames))
        self._raw_shape = [len(self._video_idx2frames)] + \
            [dummy_vid[0].shape[0]] + list(dummy_vid[0].shape[2:])

        # Apply max_size.
        self._raw_idx = np.arange(
            self._raw_shape[0], dtype=np.int64)

    @staticmethod
    def _file_ext(fname: str) -> str:
        """
        Returns the file extension of a given filename.

        Args:
            fname (str): Filename.

        Returns:
            str: File extension.
        """
        return os.path.splitext(fname)[1].lower()

    def __getitem__(self,
                    idx: int) -> dict:
        """
        Retrieves a video from the dataset at the given index.

        Args:
            idx (int): Index of the video to retrieve.

        Returns:
            dict: A dictionary containing the real-world videos.
                - 'image': Real-world video
        """
        total_len = len(self._video_idx2frames[self._raw_idx[idx]])
        random_offset = random.randint(
            0, min(self.max_num_frames, total_len) - self.num_frames - 1)
        frames_idx = np.arange(0, self.num_frames) + random_offset

        frames = self._load_raw_frames(
            self._video_idx2frames, self._raw_idx[idx], frames_idx=frames_idx)

        # Hflip all the frames together
        if self.hflip:
            if torch.rand(1) > 0.5:
                frames = TF.hflip(frames)

        return {
            'image': frames
        }

    def __len__(self) -> int:
        """
        Returns the number of videos in the dataset.

        Returns:
            int: Number of videos in the dataset.
        """
        return self._raw_idx.size

    def _load_raw_frames(self,
                         video_idx2frames: List[List[str]],
                         raw_idx: int,
                         frames_idx: List[int]) -> torch.Tensor:
        """
        Loads the raw frames for a video index and frame indices.

        Args:
            video_idx2frames (List[List[str]]): List of video frames.
            raw_idx (int): Index of the video.
            frames_idx (List[int]): List of frame indices to load.

        Returns:
            torch.Tensor: Tensor containing array of frames.
        """
        frame_paths = video_idx2frames[raw_idx]
        images = []

        frames_idx = np.array(frames_idx)

        for frame_idx in frames_idx:
            with self._open_file(frame_paths[frame_idx]) as f:
                images.append(self.load_image(f, self.transform))
        return torch.stack(images, dim=1)

    def _open_file(self, fname: str):
        """
        Opens a frame in read binary mode.

        Args:
            fname (str): File name.
        """
        return open(os.path.join(os.path.dirname(self.path), fname), 'rb')

    def load_image(self, f: IO, transform: Callable) -> torch.Tensor:
        """
        Loads a frame from a file and applies the transformation.

        Args:
            f (IO): File object.
            transform (Callable): transformation function.

        Returns:
            torch.Tensor: Transformed frame.
        """
        image = np.array(PIL.Image.open(f))
        image = transform(image)
        return image

    @property
    def image_shape(self) -> list:
        """
        Returns the shape of real-world videos.

        Returns:
            List[int]: Shape of videos.
        """
        return list(self._raw_shape[1:])


class PairedVideoDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root_path: str,
                 real_path: str,
                 cond_path: str,
                 num_frames: int,
                 max_num_frames: int = 1000000,
                 image_size: int = 64,
                 hflip: bool = False):
        """
        Dataset class for paired videos.

        Args:
            root_path (str): Root path of the dataset.
            real_path (str): Path to the real-world videos.
            cond_path (str): Path to the corresponding conditional videos.
            num_frames (int): Number of frames to sample from each video.
            max_num_frames (int, optional): Maximum number of frames to consider from each video. Defaults to 1000000.
            image_size (int, optional): Resolution of the output images. Defaults to 64.
            hflip (bool, optional): Whether to apply horizontal flipping augmentation. Defaults to False.
        """
        self.path = root_path  # Root path of the dataset
        self.real_path = real_path  # Path to the real-world videos
        self.cond_path = cond_path  # Path to the corresponding conditional videos
        self.num_frames = num_frames  # Number of frames to sample from each video
        # Maximum number of frames to consider from each video
        self.max_num_frames = max_num_frames
        self.image_size = image_size  # Resolution of the output images

        self.hflip = hflip  # Whether to apply horizontal flipping augmentation

        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize(self.image_size),
            T.CenterCrop(self.image_size),
        ])
        self.cond_transform = self.transform

        def listdir_full_paths(d: str) -> list[str]:
            """
            Returns a sorted list of full paths to files in the given directory.

            Args:
                d (str): Directory path.

            Returns:
                list[str]: Sorted list of full file paths.
            """
            return sorted([os.path.join(d, x) for x in os.listdir(d)])

        if os.path.isdir(self.real_path):
            # We assume that the depth is 2
            self._real_all_objects = {o for d in listdir_full_paths(self.real_path) for o in (
                ([d] + listdir_full_paths(d)) if os.path.isdir(d) else [d])}
            self._real_all_objects = {os.path.relpath(o, start=os.path.dirname(
                self.real_path)) for o in {self.real_path}.union(self._real_all_objects)}
        else:
            raise IOError('Path must be a directory')

        if os.path.isdir(self.cond_path):
            # We assume that the depth is 2
            self._cond_all_objects = {o for d in listdir_full_paths(self.cond_path) for o in (
                ([d] + listdir_full_paths(d)) if os.path.isdir(d) else [d])}
            self._cond_all_objects = {os.path.relpath(o, start=os.path.dirname(
                self.cond_path)) for o in {self.cond_path}.union(self._cond_all_objects)}
        else:
            raise IOError('Path must be a directory')

        PIL.Image.init()
        self._real_video_dir2frames = {}
        real_objects = sorted([d for d in self._real_all_objects])
        real_root_path_depth = len(os.path.normpath(
            real_objects[0]).split(os.path.sep))
        real_curr_d = real_objects[1]  # Root path is the first element

        for o in real_objects[1:]:
            curr_real_obj_depth = len(os.path.normpath(o).split(os.path.sep))
            if self._file_ext(o) in PIL.Image.EXTENSION:
                assert o.startswith(
                    real_curr_d), f"Object {o} is out of sync. It should lie inside {real_curr_d}"
                assert curr_real_obj_depth == real_root_path_depth + \
                    2, "Frame images should be inside directories"
                if not real_curr_d in self._real_video_dir2frames:
                    self._real_video_dir2frames[real_curr_d] = []
                self._real_video_dir2frames[real_curr_d].append(o)
            else:
                # We encountered a new directory
                assert curr_real_obj_depth == real_root_path_depth + \
                    1, f"Video directories should be inside the root dir. {o} is not."
                if real_curr_d in self._real_video_dir2frames:
                    sorted_files = sorted(
                        self._real_video_dir2frames[real_curr_d])
                    self._real_video_dir2frames[real_curr_d] = sorted_files
                real_curr_d = o
        self._real_video_idx2frames = [
            frames for frames in self._real_video_dir2frames.values()]

        if len(self._real_video_idx2frames) == 0:
            raise IOError('No videos found in the specified archive')

        self._cond_video_dir2frames = {}
        cond_objects = sorted([d for d in self._cond_all_objects])
        cond_root_path_depth = len(os.path.normpath(
            cond_objects[0]).split(os.path.sep))
        cond_curr_d = cond_objects[1]  # Root path is the first element

        for o in cond_objects[1:]:
            curr_cond_obj_depth = len(os.path.normpath(o).split(os.path.sep))
            if self._file_ext(o) in PIL.Image.EXTENSION:
                assert o.startswith(
                    cond_curr_d), f"Object {o} is out of sync. It should lie inside {cond_curr_d}"
                assert curr_cond_obj_depth == cond_root_path_depth + \
                    2, "Frame images should be inside directories"
                if not cond_curr_d in self._cond_video_dir2frames:
                    self._cond_video_dir2frames[cond_curr_d] = []
                self._cond_video_dir2frames[cond_curr_d].append(o)
            else:
                # We encountered a new directory
                assert curr_cond_obj_depth == cond_root_path_depth + \
                    1, f"Video directories should be inside the root dir. {o} is not."
                if cond_curr_d in self._cond_video_dir2frames:
                    sorted_files = sorted(
                        self._cond_video_dir2frames[cond_curr_d])
                    self._cond_video_dir2frames[cond_curr_d] = sorted_files
                cond_curr_d = o
        self._cond_video_idx2frames = [
            frames for frames in self._cond_video_dir2frames.values()]

        if len(self._cond_video_idx2frames) == 0:
            raise IOError('No videos found in the specified archive')

        dummy_vid = self._load_raw_frames(
            self._real_video_idx2frames, self._cond_video_idx2frames, 0, np.arange(num_frames))
        self._raw_shape = [len(self._real_video_idx2frames)] + \
            [dummy_vid[0].shape[0]] + list(dummy_vid[0].shape[2:])
        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)

    @staticmethod
    def _file_ext(fname: str) -> str:
        """
        Returns the file extension of a given filename.

        Args:
            fname (str): Filename.

        Returns:
            str: File extension.
        """
        return os.path.splitext(fname)[1].lower()

    def __getitem__(self,
                    idx: int) -> dict:
        """
        Retrieves a pair of videos from the dataset at the given index.

        Args:
            idx (int): Index of the video to retrieve.

        Returns:
            dict: A dictionary containing the real-world and corresponding conditional videos.
                - 'image': Real-world video
                - 'cond': Corresponding Conditional video
        """
        real_total_len = len(self._real_video_idx2frames[self._raw_idx[idx]])

        cond_total_len = len(
            self._cond_video_idx2frames[self._raw_idx[idx]])

        total_len = min(real_total_len, cond_total_len)
        random_offset = random.randint(
            0, min(self.max_num_frames, total_len) - self.num_frames - 1)

        frames_idx = np.arange(0, self.num_frames) + random_offset

        real_frames, cond_frames = self._load_raw_frames(
            self._real_video_idx2frames, self._cond_video_idx2frames, self._raw_idx[idx], frames_idx=frames_idx)

        # Hflip all the frames together
        if self.hflip:
            if torch.rand(1) > 0.5:
                real_frames = TF.hflip(real_frames)
                cond_frames = TF.hflip(cond_frames)

        return {
            'image': real_frames,
            'cond': cond_frames
        }

    def __len__(self) -> int:
        """
        Returns the number of videos in the dataset.

        Returns:
            int: Number of videos in the dataset.
        """
        return self._raw_idx.size

    def _load_raw_frames(
            self,
            real_video_idx2frames: List[List[str]],
            cond_video_idx2frames: List[List[str]],
            raw_idx: int,
            frames_idx: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads the raw frames for a pair videos index and frame indices.

        Args:
            real_video_idx2frames (List[List[str]]): List of real video frames.
            cond_video_idx2frames (List[List[str]]): List of corresponding conditional video frames.
            raw_idx (int): Index of the video.
            frames_idx (List[int]): List of frame indices to load.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing arrays of real frames and corresponding conditional frames.
        """

        real_frame_paths = real_video_idx2frames[raw_idx]
        real_images = []

        cond_frame_paths = cond_video_idx2frames[raw_idx]
        depth_images = []

        frames_idx = np.array(frames_idx)

        for frame_idx in frames_idx:
            with self._open_file(real_frame_paths[frame_idx]) as f:
                real_images.append(self.load_image(f, self.transform))

            with self._open_file(cond_frame_paths[frame_idx]) as f:
                depth_images.append(self.load_image(f, self.cond_transform))

        return [torch.stack(real_images, dim=1), torch.stack(depth_images, dim=1)]

    def _open_file(self, fname: str):
        """
        Opens a frame in read binary mode.

        Args:
            fname (str): File name.
        """
        return open(os.path.join(self.path, fname), 'rb')

    def load_image(self, f: IO, transform: Callable) -> torch.Tensor:
        """
        Loads a frame from a file and applies the transformation.

        Args:
            f (IO): File object.
            transform (Callable): transformation function.

        Returns:
            torch.Tensor: Transformed frame.
        """
        image = np.array(PIL.Image.open(f))
        image = transform(image)
        return image

    @property
    def cond_image_shape(self) -> List[int]:
        """
        Returns the shape of corresponding conditional videos.

        Returns:
            List[int]: Shape of corresponding conditional videos.
        """
        return list(self._raw_shape[1:])

    @property
    def image_shape(self) -> List[int]:
        """
        Returns the shape of real-world videos.

        Returns:
            List[int]: Shape of videos.
        """
        return list(self._raw_shape[1:])


class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        shuffle: bool = True,
        seed: int = 0,
        window_size: float = 0.5
    ):
        """
        Infinite sampler for a dataset.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to sample from.
            shuffle (bool, optional): Whether to shuffle the indices. Defaults to True.
            seed (int, optional): Seed value for the random number generator. Defaults to 0.
            window_size (float, optional): Proportion of the dataset used as a window for sampling. Defaults to 0.5.
        """
        assert len(dataset) > 0
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self) -> Iterator[int]:
        """
        Returns an iterator over the indices of the dataset.

        Returns:
            Iterator[int]: Iterator over the indices.
        """
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1

    def __len__(self) -> int:
        """
        Returns the number of samples in the sampler.

        Returns:
            int: Number of samples in the sampler.
        """
        # The sampler does not have a fixed length as it is infinite
        return float('inf')
