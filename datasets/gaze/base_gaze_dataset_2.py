# from typing import Sequence
# import torch
# import random
# import os
# import numpy as np
# import cv2
# from omegaconf import DictConfig
# from torchvision import transforms
# from pathlib import Path
# from abc import abstractmethod, ABC
# import json
# import pandas as pd
# from PIL import Image


# class BaseGazeDataset(torch.utils.data.Dataset, ABC):
#     """
#     Base class for gaze-based video datasets. Each sample is a sequence of frames
#     aligned with gaze positions loaded from per-user .txt files.

#     Folder structure (example):
#     - [save_dir] (from cfg.save_dir)
#         - /filtered_gaze_data
#             - v01_Hugo_2172_u007_f_23_1_fwd.txt
#             - ...
#         - /SAVAM
#             - /v01_Hugo_2172/
#                 - 0001.png
#                 - 0002.png
#                 - ...
#     """

#     def __init__(self, cfg: DictConfig, split: str = "training"):
#         super().__init__()
#         self.cfg = cfg

#         self.split = split
#         self.resolution = cfg.resolution
#         self.sequence_length = cfg.sequence_length
#         self.frame_skip = cfg.frame_skip
#         self.validation_multiplier = getattr(cfg, "validation_multiplier", 1)
#         self.stride = getattr(cfg, "stride", 1)
#         self.external_cond_dim = getattr(cfg, "external_cond_dim", 0)
#         self.save_dir = Path(cfg.save_dir)
#         self.save_dir.mkdir(exist_ok=True, parents=True)
        
#         # main dirs
#         self.gaze_dir = self.save_dir / f"{cfg.gaze_type}_gaze_data"
#         self.frame_dir = self.save_dir / cfg.frame_dir
#         self.metadata_path = self.save_dir / "metadata.json"

#         # if metadata doesn't exist, trigger dataset setup
#         if not self.metadata_path.exists():
#             print(f"Creating dataset metadata in {self.save_dir}...")
#             self.download_dataset()
#             json.dump(
#                 {
#                     "training": self.get_data_lengths("training"),
#                     "validation": self.get_data_lengths("validation"),
#                 },
#                 open(self.metadata_path, "w"),
#             )

#         self.metadata = json.load(open(self.metadata_path, "r"))
#         self.data_paths = self.get_data_paths(split)
#         self.clips_per_gaze = np.clip(
#             np.array(self.metadata[split]) - self.sequence_length + 1, a_min=1, a_max=None
#         ).astype(np.int32)
#         self.cum_clips_per_gaze = np.cumsum(self.clips_per_gaze)
#         self.transform = transforms.Compose([
#             transforms.Resize((self.resolution, self.resolution), antialias=True),
#             transforms.ToTensor(),
#         ])

#         # deterministic shuffle
#         random.seed(0)
#         self.idx_remap = list(range(self.__len__()))
#         random.shuffle(self.idx_remap)

#     # ----------------------------------------------------------------------
#     # Abstract methods (must be implemented in inheritors)
#     # ----------------------------------------------------------------------

#     @abstractmethod
#     def download_dataset(self) -> Sequence[int]:
#         """Download or prepare dataset in save_dir"""
#         raise NotImplementedError

#     @abstractmethod
#     def get_data_paths(self, split):
#         """Return list of gaze .txt file paths for the split"""
#         raise NotImplementedError

#     # ----------------------------------------------------------------------
#     # Metadata and indexing
#     # ----------------------------------------------------------------------

#     def get_data_lengths(self, split):
#         """
#         Return number of valid frame-gaze pairs per file.
#         Each .txt file is parsed to count how many valid gaze entries can form sequences.
#         """
#         lengths = []
#         for path in self.get_data_paths(split):
#             try:
#                 df = pd.read_csv(
#                     path,
#                     sep=r"\s+",
#                     header=None,
#                     names=["timestamp", "left_x", "left_y", "right_x", "right_y"],
#                 )
#                 df = df[(df != 0).all(axis=1)]
#                 lengths.append(len(df))
#             except Exception:
#                 lengths.append(0)
#         return lengths

#     def split_idx(self, idx):
#         """Map a global index into (gaze_file_idx, local_sequence_start_idx)."""
#         gaze_idx = np.argmax(self.cum_clips_per_gaze > idx)
#         seq_idx = idx - np.pad(self.cum_clips_per_gaze, (1, 0))[gaze_idx]
#         return gaze_idx, seq_idx

#     # ----------------------------------------------------------------------
#     # Data loading utilities
#     # ----------------------------------------------------------------------

#     @staticmethod
#     def load_image(filename: Path):
#         """Load an image as RGB (C,H,W)"""
#         image = cv2.imread(str(filename))
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         return np.transpose(image, (2, 0, 1))

#     @staticmethod
#     def normalize_gaze(gaze_data, image_dimensions):
#         """Normalize gaze coordinates to [0,1]"""
#         gaze_data[:, [1, 3]] /= image_dimensions[0]
#         gaze_data[:, [2, 4]] /= image_dimensions[1]
#         return gaze_data

#     @staticmethod
#     def align_gaze_to_frames(gaze_data, frame_dir):
#         """Align gaze timestamps to available frame files (assuming 40ms step)."""
#         first_ts = gaze_data[0, 0]
#         frame_files = sorted(
#             [f for f in os.listdir(frame_dir) if f.endswith(".png")],
#             key=lambda x: int(os.path.splitext(x)[0]),
#         )
#         frame_map = {int(os.path.splitext(f)[0]): f for f in frame_files}

#         aligned = []
#         for row in gaze_data:
#             ts, lx, ly, rx, ry = row
#             rel_frame = 1 + int((ts - first_ts) / 40000)
#             if rel_frame in frame_map:
#                 aligned.append(((lx, ly, rx, ry), str(Path(frame_dir) / frame_map[rel_frame])))
#         return aligned

#     # ----------------------------------------------------------------------
#     # PyTorch dataset interface
#     # ----------------------------------------------------------------------

#     def __len__(self):
#         return self.clips_per_gaze.sum()

#     def __getitem__(self, idx):
#         idx = self.idx_remap[idx]
#         gaze_idx, seq_idx = self.split_idx(idx)
#         gaze_path = self.data_paths[gaze_idx]

#         df = pd.read_csv(
#             gaze_path,
#             sep=r"\s+",
#             header=None,
#             names=["timestamp", "left_x", "left_y", "right_x", "right_y"],
#         )
#         df = df[(df != 0).all(axis=1)]
#         gaze_data = df.to_numpy()
#         gaze_data = self.normalize_gaze(gaze_data, (1920,1080))

#         videoname = gaze_path.stem.split("_")[0]
#         frame_dirs = [d for d in os.listdir(self.frame_dir) if d.startswith(videoname)]
#         if not frame_dirs:
#             raise FileNotFoundError(f"No frame directory for {videoname}")
#         frame_dir = self.frame_dir / frame_dirs[0]

#         aligned = self.align_gaze_to_frames(gaze_data, frame_dir)
#         start = seq_idx
#         end = start + self.sequence_length

#         if end > len(aligned):
#             pad_len = end - len(aligned)
#             aligned += [aligned[-1]] * pad_len  # repeat last entry
#         else:
#             pad_len = 0

#         seq_gaze = [x[0] for x in aligned[start:end]]
#         seq_frames = [x[1] for x in aligned[start:end]]

#         # Load and transform frames
#         video = []
#         for f in seq_frames:
#             img = Image.open(f).convert("RGB")
#             img = self.transform(img)
#             video.append(img)
#         video = torch.stack(video, dim=0)  # (T,C,H,W)

#         nonterminal = np.ones(self.sequence_length)
#         if pad_len > 0:
#             nonterminal[-pad_len:] = 0

#         gaze_tensor = torch.tensor(seq_gaze, dtype=torch.float32)
#         return video[:: self.frame_skip], gaze_tensor[:: self.frame_skip], nonterminal[:: self.frame_skip]
