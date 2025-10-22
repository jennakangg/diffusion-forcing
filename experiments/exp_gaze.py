from datasets.gaze import (
    Ego4DGazeDataset,
)
from algorithms.diffusion_forcing import DiffusionForcingScanpath
from .exp_base import BaseLightningExperiment


class GazePredictionExperiment(BaseLightningExperiment):
    """
    A video prediction experiment
    """

    compatible_algorithms = dict(
        df_gaze=DiffusionForcingScanpath,
    )

    compatible_datasets = dict(
        # video datasets
        gaze=Ego4DGazeDataset,
    )
