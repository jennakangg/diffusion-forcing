# import os
# import torch
# import pandas as pd
# import numpy as np
# from tqdm import tqdm
# from pathlib import Path
# from typing import Sequence
# from .base_gaze_dataset import BaseGazeDataset
# import random

# class SavamGazeDataset(BaseGazeDataset):
#     """
#     SAVAM Gaze Dataset
#     - Gaze .txt per user recording
#     - Frame directories containing extracted PNGs
#     - Supports normalized gaze alignment and sequential slicing
#     """

#     def __init__(self, cfg, split="training"):
#         self.cfg = cfg
#         self.split = split
#         self.split_ratios = getattr(cfg, "split_ratios", (0.8, 0.1, 0.1))  # train, val, test
#         super().__init__(cfg, split)

#     def download_dataset(self) -> Sequence[int]:
#         """No download step for local gaze datasets."""
#         print("SAVAM dataset assumed to be pre-downloaded and extracted.")
#         return []

#     def get_data_paths(self, split):
#         all_files = sorted(
#             [self.gaze_dir / f for f in os.listdir(self.gaze_dir) if f.endswith(".txt")]
#         )

#         # Deterministic split
#         random.seed(42)
#         random.shuffle(all_files)

#         n_total = len(all_files)
#         n_train = int(self.split_ratios[0] * n_total)
#         n_val = int(self.split_ratios[1] * n_total)

#         if split == "training":
#             return all_files[:n_train]
#         elif split == "validation":
#             return all_files[n_train:n_train + n_val]
#         elif split == "test":
#             return all_files[n_train + n_val:]
#         else:
#             raise ValueError(f"Unknown split '{split}'")

#     def _load_sequences(self, debug=False):
#         samples = []
#         gaze_files = self.get_data_paths(self.split)
#         if debug:
#             gaze_files = gaze_files[:100]

#         for file_path in tqdm(gaze_files, desc=f"Loading gaze sequences ({self.split})"):
#             df = pd.read_csv(
#                 file_path,
#                 sep=r"\s+",
#                 header=None,
#                 names=["timestamp", "left_x", "left_y", "right_x", "right_y"]
#             )
#             df = df[(df != 0).all(axis=1)]
#             if len(df) < self.sequence_length + 1:
#                 continue

#             gaze_data = df.to_numpy()
#             gaze_data = self._normalize_gaze(gaze_data)
#             videoname = file_path.stem.split('_')[0]

#             # Find frame folder for this video
#             frame_dirs = [d for d in os.listdir(self.frame_path) if d.startswith(videoname)]
#             if not frame_dirs:
#                 continue
#             frame_dir = self.frame_path / frame_dirs[0]
#             frame_files = sorted(
#                 [f for f in os.listdir(frame_dir) if f.endswith(".png")],
#                 key=lambda x: int(os.path.splitext(x)[0])
#             )
#             frame_map = {int(os.path.splitext(f)[0]): f for f in frame_files}

#             first_timestamp = gaze_data[0, 0]
#             aligned = []
#             for row in gaze_data:
#                 timestamp, lx, ly, rx, ry = row
#                 rel_frame = 1 + int((timestamp - first_timestamp) / 40000)
#                 if rel_frame in frame_map:
#                     aligned.append(((lx, ly, rx, ry), str(frame_dir / frame_map[rel_frame])))

#             # Slice into fixed-length sequences
#             for i in range(0, len(aligned) - self.sequence_length - 1, self.stride):
#                 input_seq = [x[0] for x in aligned[i:i + self.sequence_length]]
#                 target = aligned[i + self.sequence_length][0]
#                 frame_seq = [x[1] for x in aligned[i:i + self.sequence_length]]
#                 samples.append((
#                     torch.tensor(input_seq, dtype=torch.float32),
#                     torch.tensor(target, dtype=torch.float32),
#                     frame_seq
#                 ))

#         return samples
