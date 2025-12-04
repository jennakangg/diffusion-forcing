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
        # === Initialize and cache ResNet backbone once ===
        model_name = getattr(cfg, "resnet_model", "resnet18")

        try:
            weights = getattr(models, model_name).weights.DEFAULT
        except AttributeError:
            # For backward compatibility if DEFAULT not defined
            weights = "IMAGENET1K_V1"

        backbone = getattr(models, model_name)(weights=weights)
        self.resnet = torch.nn.Sequential(*list(backbone.children())[:-2]).eval()

        if torch.cuda.is_available():
            self.resnet = self.resnet.cuda()

        self.resnet_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

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
        raw_clip = gaze_points[frame_idx:end_idx].copy()  # (T,2)

        nonterminal = np.ones(self.n_frames)

        # normalize if needed
        clip = clip / np.array([[self.cfg.dataset_video_resolution, self.cfg.dataset_video_resolution]])

        # convert to tensor
        clip = torch.from_numpy(clip).float()  # (T, 2)

        # pad to 3 channels (x, y, dummy)
        clip = F.pad(clip, (0, 1), mode="constant", value=0.0)  # (T, 3)

        # add spatial dims
        clip = clip.unsqueeze(-1).unsqueeze(-1)      # (T, 3, 1, 1)
        clip = clip.repeat(1, 1, self.cfg.resolution, self.cfg.resolution)

        clip = clip.contiguous()

        # # === Load only the first frame of this clip ===
        # cap = cv2.VideoCapture(video_path)
        # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        # ret, frame = cap.read()
        # cap.release()
        # if not ret:
        #     raise ValueError(f"Failed to read frame {frame_idx} from {video_path}")

        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        # # === Transform and extract ResNet features ===
        # frame_tensor = self.resnet_transform(frame_rgb).unsqueeze(0)  # (1,3,224,224)
        # if torch.cuda.is_available():
        #     frame_tensor = frame_tensor.cuda()
        #     with torch.no_grad():
        #         feat_map = self.resnet(frame_tensor).cpu()  # (1,C,H',W')
        # else:
        #     with torch.no_grad():
        #         feat_map = self.resnet(frame_tensor)
        

        # # === Flatten spatial dims to get action condition ===
        # feat_flat = F.adaptive_avg_pool2d(feat_map, (1, 1)).squeeze().float()  # (C,)
        # action_condition = feat_flat.unsqueeze(0).repeat(clip[::self.frame_skip].shape[0], 1)

        # ===============================
        # Load frames every K steps
        # ===============================
        K = 10   # condition every 10 frames

        Tprime = clip[::self.frame_skip].shape[0]   # number of model timesteps
        cond_features = []

        cap = cv2.VideoCapture(video_path)

        for t in range(0, Tprime):
            # map model timestep → actual video frame index
            vid_frame_idx = frame_idx + t * self.frame_skip

            # condition only every K frames
            if t % K == 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, vid_frame_idx)
                ret, frame = cap.read()
                if not ret:
                    # fallback to previous feature (or zeros)
                    if len(cond_features) > 0:
                        cond_features.append(cond_features[-1])
                    else:
                        cond_features.append(torch.zeros(512))
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # ---- Extract ResNet features ----
                frame_tensor = self.resnet_transform(frame_rgb).unsqueeze(0)
                if torch.cuda.is_available():
                    frame_tensor = frame_tensor.cuda()

                with torch.no_grad():
                    fmap = self.resnet(frame_tensor)  # (1, C, H', W')
                    feat_flat = F.adaptive_avg_pool2d(fmap, (1, 1)).view(-1).cpu()

                cond_features.append(feat_flat)
            else:
                # repeat last feature until next K-step feature arrives
                cond_features.append(cond_features[-1])

        cap.release()

        # final tensor shape = (T', C)
        action_condition = torch.stack(cond_features, dim=0).float()
        # print("dataset_video_resolution", self.cfg.dataset_video_resolution, raw_clip[0], clip[0,0,0,0]*self.cfg.dataset_video_resolution)
        # === Return tuple ===
        return (
            clip[:: self.frame_skip],      # (T', 3, res, res)
            action_condition,                   # (T', 512)
            torch.from_numpy(nonterminal[:: self.frame_skip]).float(),
            video_path,
            int(frame_idx),
            torch.from_numpy(raw_clip[:: self.frame_skip]).float()
        )