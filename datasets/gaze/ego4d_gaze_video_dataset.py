import json
from pathlib import Path
import pandas as pd
from typing import Sequence
from .base_gaze_dataset import BaseGazeDataset
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
import cv2
import numpy as np
import os
import glob
from pathlib import Path

class Ego4DGazeVideoDataset(BaseGazeDataset):
    """
    Concrete implementation of BaseGaze2DDataset for Ego4D-style 2D gaze CSVs.

    Folder layout (after preprocessing):
        data_split/
        ├── training/
        │   ├── <video_id>_general_eye_gaze_2d.csv
        └── validation/
            ├── <video_id>_general_eye_gaze_2d.csv
    """

    def __init__(self, cfg, split="training"):
        super().__init__(cfg, split)

    def download_dataset(self) -> Sequence[int]:
        """
        This dataset is assumed to be pre-downloaded and organized into
        training/ and validation/ directories.

        We just compute and return the lengths of each CSV file to build metadata.json.
        """
        print("Verifying existing Ego4D gaze dataset structure...")

        train_paths = list((self.save_dir / "training").glob("*_general_eye_gaze_2d.csv"))
        val_paths = list((self.save_dir / "validation").glob("*_general_eye_gaze_2d.csv"))

        if not train_paths and not val_paths:
            raise FileNotFoundError(
                f"No gaze CSVs found in {self.save_dir}/training or /validation./n"
                "Please run your preprocessing/splitting script first."
            )

        lengths = {
            "training": self.get_data_lengths("training"),
            "validation": self.get_data_lengths("validation"),
        }

        print(
            f"Found {len(lengths['training'])} training files "
            f"and {len(lengths['validation'])} validation files."
        )
        return lengths

    
    def __getitem__(self, idx):
        """
        Loads a gaze clip and conditions it on the first video frame's ResNet features.
        Gaze CSV: [take_name]_general_eye_gaze_2d.csv
        Video: takes/[take_name]/frame_aligned_videos/aria*.mp4
        """
        idx = self.idx_remap[idx]
        file_idx, frame_idx = self.split_idx(idx)
        gaze_path = self.data_paths[file_idx]
        gaze_points = self.load_gaze_points(gaze_path)  # (T, 2)

        # === Clean invalid values ===
        valid_mask = ~np.isnan(gaze_points).any(axis=1) & ~np.isinf(gaze_points).any(axis=1)
        gaze_points = gaze_points[valid_mask]
        if len(gaze_points) == 0:
            raise ValueError(f"No valid gaze data in {gaze_path}")

        # === Slice ===
        end_idx = frame_idx + self.frame_stride * self.n_frames
        if end_idx > len(gaze_points):
            new_idx = (idx + 1) % len(self.idx_remap)
            return self.__getitem__(new_idx)

        clip_gaze = gaze_points[frame_idx:end_idx:self.frame_stride]
        clip_gaze = clip_gaze / np.array([[self.cfg.dataset_video_resolution, self.cfg.dataset_video_resolution]])
        clip_gaze = torch.from_numpy(clip_gaze).float()
        clip_gaze = F.pad(clip_gaze, (0, 1), mode="constant", value=0.0)
        clip_gaze = clip_gaze.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.cfg.resolution, self.cfg.resolution)

        # === Find matching video ===
        video_prefix = os.path.basename(gaze_path).replace("_general_eye_gaze_2d.csv", "")
        take_dir = Path(self.cfg.takes_root) / video_prefix / "frame_aligned_videos"
        video_files = glob.glob(str(take_dir / "aria*.mp4"))
        if len(video_files) == 0:
            raise FileNotFoundError(f"No video found for {video_prefix}")
        video_path = video_files[0]

        # === Load only the first frame of this clip ===
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError(f"Failed to read frame {frame_idx} from {video_path}")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # === Prepare and cache ResNet backbone ===
        if not hasattr(self, "resnet"):
            model_name = getattr(self.cfg, "resnet_model", "resnet18")
            backbone = getattr(models, model_name)(weights="IMAGENET1K_V1")
            self.resnet = torch.nn.Sequential(*list(backbone.children())[:-2]).eval()  # drop avgpool & fc
            self.resnet.cuda() if torch.cuda.is_available() else None
            self.resnet_transform = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])

        # === Transform and extract ResNet features ===
        frame_tensor = self.resnet_transform(frame_rgb).unsqueeze(0)  # (1,3,224,224)
        if torch.cuda.is_available():
            frame_tensor = frame_tensor.cuda()
            with torch.no_grad():
                feat_map = self.resnet(frame_tensor).cpu()  # (1,C,H',W')
        else:
            with torch.no_grad():
                feat_map = self.resnet(frame_tensor)

        # === Flatten spatial dims to get action condition ===
        feat_flat = F.adaptive_avg_pool2d(feat_map, (1, 1)).squeeze().float()  # (C,)
        action_condition = feat_flat.unsqueeze(0).repeat(clip_gaze[::self.frame_skip].shape[0], 1)


        # === Nonterminal ===
        nonterminal = torch.ones(clip_gaze.shape[0])

        # === Return tuple ===
        return (
            clip_gaze[:: self.frame_skip],      # (T', 3, res, res)
            action_condition,                   # (T', 512)
            nonterminal,                        # (T',)
        )