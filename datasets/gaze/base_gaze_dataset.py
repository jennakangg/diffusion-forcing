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
import matplotlib.pyplot as plt
import glob


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
        self.plot_counter = 0 
        self.debug_dir = "debug/ego4d"

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

        df = df.dropna(subset=['x', 'y'])

        # auto-detect columns
        if 'x' in df.columns and 'y' in df.columns:
            coords = df[['x', 'y']].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        else:
            print(f"NO XY COL: {csv_path}")
            coords = df.iloc[:, -2:].to_numpy(dtype=np.float32)
        return coords  # shape: (T, 2)

    def __len__(self):
        return self.clips_per_video.sum()

    def __getitem__(self, idx):
        idx = self.idx_remap[idx]
        file_idx, frame_idx = self.split_idx(idx)
        gaze_path = self.data_paths[file_idx]
        gaze_points = self.load_gaze_points(gaze_path)  # (T, 2)
        gaze_points = np.nan_to_num(gaze_points, nan=0.0, posinf=0.0, neginf=0.0)

        n_nans = np.isnan(gaze_points).sum()
        n_infs = np.isinf(gaze_points).sum()
        if n_nans > 0 or n_infs > 0:
            print(f"[WARN] {gaze_path}: NaNs={n_nans}, Infs={n_infs}")
            # Clean values
            valid_mask = ~np.isnan(gaze_points).any(axis=1) & ~np.isinf(gaze_points).any(axis=1)
            n_removed = len(gaze_points) - valid_mask.sum()
            print(f"   → Removed {n_removed} invalid rows")
            gaze_points = gaze_points[valid_mask]

        # slice
        # clip = gaze_points[frame_idx : frame_idx + self.n_frames]

        # === Find matching video ===
        video_prefix = os.path.basename(gaze_path).replace("_general_eye_gaze_2d.csv", "")
        take_dir = Path(self.cfg.takes_root) / video_prefix / "frame_aligned_videos"
        video_files = glob.glob(str(take_dir / "aria*214-1.mp4"))

        if len(video_files) == 0:
            print(take_dir)
            raise FileNotFoundError(f"No video found for {video_prefix}")
        video_path = video_files[0]

        end_idx = frame_idx + self.frame_skip * self.n_frames

        end_idx = frame_idx + self.frame_skip * self.n_frames
        if end_idx > len(gaze_points):
            # not enough data left for a full clip → skip
            new_idx = (idx + 1) % len(self.idx_remap)
            return self.__getitem__(new_idx)
        
        
        # normal slicing (only valid length clips)
        clip = gaze_points[frame_idx:end_idx]

        nonterminal = np.ones(self.n_frames)
        raw_clip = gaze_points[frame_idx:end_idx].copy()  # (T,2)

        # normalize if needed
        clip = clip / self.cfg.dataset_video_resolution

        # convert to tensor
        clip = torch.from_numpy(clip).float()  # (T, 2)

        # pad to 3 channels (x, y, dummy)
        clip = F.pad(clip, (0, 1), mode="constant", value=0.0)  # (T, 3)

        # add spatial dims
        clip = clip.unsqueeze(-1).unsqueeze(-1)      # (T, 3, 1, 1)
        clip = clip.repeat(1, 1, self.cfg.resolution, self.cfg.resolution)

        clip = clip.contiguous()
        
        T_prime = clip[:: self.frame_skip].shape[0]

        abs_video_idx = np.arange(T_prime) * (self.frame_skip) + frame_idx

        print(clip[:: self.frame_skip].shape)


        return (
            clip[:: self.frame_skip],
            torch.zeros((T_prime,)),
            torch.from_numpy(nonterminal[:: self.frame_skip]).float(),
            video_path,
            torch.from_numpy(abs_video_idx).long(),
            torch.from_numpy(raw_clip[:: self.frame_skip]).float(),  # NEW: raw gaze for debugging
        )



