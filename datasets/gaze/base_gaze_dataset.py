import torch
import random
import os
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from pathlib import Path
from abc import abstractmethod, ABC
import json
import torch.nn.functional as F


class BaseGazeDataset(torch.utils.data.Dataset, ABC):
    """
    Base class for 2D gaze datasets.
    Dataset folder structure:
    - [save_dir]/
        - training/
            - [video_id]_general_eye_gaze_2d.csv
        - validation/
            - [video_id]_general_eye_gaze_2d.csv
        metadata.json
    """

    def __init__(self, cfg: DictConfig, split: str = "training"):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.save_dir = Path(cfg.save_dir)
        self.split_dir = self.save_dir / split
        self.save_dir.mkdir(exist_ok=True, parents=True)

        self.n_frames = (
            cfg.n_frames * cfg.frame_skip
            if split == "training"
            else cfg.n_frames * cfg.frame_skip * cfg.validation_multiplier
        )
        self.frame_skip = cfg.frame_skip

        self.metadata_path = self.save_dir / "metadata.json"
        if not self.metadata_path.exists():
            print(f"Creating dataset metadata in {self.save_dir}...")
            self.download_dataset()
            json.dump(
                {
                    "training": self.get_data_lengths("training"),
                    "validation": self.get_data_lengths("validation"),
                },
                open(self.metadata_path, "w"),
            )

        self.metadata = json.load(open(self.metadata_path, "r"))
        self.data_paths = self.get_data_paths(split)
        self.clips_per_video = np.clip(np.array(self.metadata[split]) - self.n_frames + 1, a_min=1, a_max=None).astype(np.int32)
        self.cum_clips_per_video = np.cumsum(self.clips_per_video)

        # deterministic shuffle
        random.seed(0)
        self.idx_remap = list(range(self.__len__()))
        random.shuffle(self.idx_remap)

    @abstractmethod
    def download_dataset(self):
        """Optionally implement download/preprocessing if needed."""
        raise NotImplementedError

    def get_data_paths(self, split):
        """Return a list of CSV paths for the given split."""
        return sorted(list((self.save_dir / split).glob("*_general_eye_gaze_2d.csv")))

    def get_data_lengths(self, split):
        """Return number of gaze samples per CSV."""
        lengths = []
        for path in self.get_data_paths(split):
            df = pd.read_csv(path)
            lengths.append(len(df))
        return lengths

    def split_idx(self, idx):
        video_idx = np.argmax(self.cum_clips_per_video > idx)
        frame_idx = idx - np.pad(self.cum_clips_per_video, (1, 0))[video_idx]
        return video_idx, frame_idx

    @staticmethod
    def load_gaze_points(csv_path: Path):
        """
        Load 2D gaze coordinates (x, y) from a CSV file.
        Expected columns: ['frame_idx', 'x', 'y'] or ['x', 'y'].
        """
        df = pd.read_csv(csv_path)
        # auto-detect columns
        if 'x' in df.columns and 'y' in df.columns:
            coords = df[['x', 'y']].to_numpy(dtype=np.float32)
        else:
            coords = df.iloc[:, -2:].to_numpy(dtype=np.float32)
        return coords  # shape: (T, 2)

    def __len__(self):
        return self.clips_per_video.sum()

    def __getitem__(self, idx):
        idx = self.idx_remap[idx]
        file_idx, frame_idx = self.split_idx(idx)
        gaze_path = self.data_paths[file_idx]
        gaze_points = self.load_gaze_points(gaze_path)  # (T, 2)

        # slice
        clip = gaze_points[frame_idx : frame_idx + self.n_frames]
        pad_len = self.n_frames - len(clip)
        
        nonterminal = np.ones(self.n_frames)
        if len(clip) < self.n_frames:
            clip = np.pad(clip, ((0, pad_len), (0, 0)))
            nonterminal[-pad_len:] = 0

        # normalize if needed
        # if np.max(clip) > 1.0:
        #     clip = clip / np.array([[self.cfg.screen_width, self.cfg.screen_height]])

        # convert to tensor
        clip = torch.from_numpy(clip).float()  # (T, 2)

        # pad to 3 channels (x, y, dummy)
        clip = F.pad(clip, (0, 1), mode="constant", value=0.0)  # (T, 3)

        # add spatial dims
        clip = clip.unsqueeze(-1).unsqueeze(-1)      # (T, 3, 1, 1)
        clip = clip.repeat(1, 1, self.cfg.resolution, self.cfg.resolution)             # (T, 3, 32, 32)
        # === critical part ===
        # keep (T, C, H, W) to match video version
        clip = clip.contiguous()

        return (
            clip[:: self.frame_skip],                         # (T', 3, 1, 1)
            torch.zeros((clip[:: self.frame_skip].shape[0],)), # dummy actions if needed
            torch.from_numpy(nonterminal[:: self.frame_skip]).float(),
        )


