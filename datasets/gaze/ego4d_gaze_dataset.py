import json
from pathlib import Path
import pandas as pd
from typing import Sequence
from .base_gaze_dataset import BaseGazeDataset
class Ego4DGazeDataset(BaseGazeDataset):
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
                f"No gaze CSVs found in {self.save_dir}/training or /validation.\n"
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
