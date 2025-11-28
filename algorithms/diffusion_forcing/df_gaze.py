from omegaconf import DictConfig
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from algorithms.common.metrics import (
    FrechetInceptionDistance,
    LearnedPerceptualImagePatchSimilarity,
    FrechetVideoDistance,
)
from .df_base import DiffusionForcingBase
from utils.logging_utils import log_video, get_validation_metrics_for_simple, log_gaze_video_2d, log_real_video_with_gaze, load_raw_frames


class DiffusionForcingScanpath(DiffusionForcingBase):
    """
    A scanpath prediction algorithm using Diffusion Forcing.
    """

    def __init__(self, cfg: DictConfig):
        self.metrics = cfg.metrics
        self.n_tokens = cfg.n_frames // cfg.frame_stack  # number of max tokens for the model
        super().__init__(cfg)

    def _build_model(self):
        super()._build_model()
        self.validation_fid_model = FrechetInceptionDistance(feature=64) if "fid" in self.metrics else None
        self.validation_lpips_model = LearnedPerceptualImagePatchSimilarity() if "lpips" in self.metrics else None
        self.validation_fvd_model = [FrechetVideoDistance()] if "fvd" in self.metrics else None

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        output_dict = super().training_step(batch, batch_idx)
        # log the video
        if batch_idx % 5000 == 0 and self.logger:
            log_video(
                output_dict["xs_pred"],
                output_dict["xs"],
                step=self.global_step,
                namespace="training_vis",
                logger=self.logger.experiment,
            )
            # log_gaze_video_2d(
            #     output_dict["xs_pred"],
            #     output_dict["xs"],
            #     step=self.global_step,
            #     namespace="training_vis_2d",
            #     resolution=self.cfg.dataset_video_resolution,
            #     logger=self.logger.experiment,
            # )
            log_real_video_with_gaze(
                output_dict["xs_pred"],
                output_dict["xs"],
                raw_gaze=output_dict["raw_gaze"],
                video_paths=output_dict["video_paths"],
                start_idxs=output_dict["start_idxs"],
                logger=self.logger,
                namespace="training_vis",
                step=self.global_step,
                resolution=self.cfg.dataset_video_resolution,
                frame_stride=self.cfg.frame_stride,
            )
            
        return output_dict

    def on_validation_epoch_end(self, namespace="validation") -> None:
        if not self.validation_step_outputs:
            return

        xs_pred, xs, video_paths, start_idxs, raw_gaze = zip(*self.validation_step_outputs)

        xs_pred = torch.cat(xs_pred, dim=1)
        xs = torch.cat(xs, dim=1)

        # === optional visualization logging ===
        if self.logger:
            log_video(
                xs_pred,
                xs,
                step=None if namespace == "test" else self.global_step,
                namespace=f"{namespace}_vis",
                context_frames=self.context_frames,
                logger=self.logger.experiment,
            )

            # log_gaze_video_2d(
            #     pred=xs_pred,
            #     gt=xs,
            #     step=None if namespace == "test" else self.global_step,
            #     namespace=f"{namespace}_vis_2d",
            #     resolution=self.cfg.dataset_video_resolution,
            #     logger=self.logger.experiment,
            # )

            log_real_video_with_gaze(
                pred=xs_pred,
                gt=xs,
                raw_gaze=raw_gaze,
                video_paths=video_paths,
                start_idxs=start_idxs,
                logger=self.logger,
                namespace=namespace,
                step=self.global_step,
                resolution=self.cfg.dataset_video_resolution,
                frame_stride=self.cfg.frame_stride,
            )


        # === compute simple metrics: MSE, PSNR, SSIM ===
        metric_dict = get_validation_metrics_for_simple(
            xs_pred[self.context_frames :],
            xs[self.context_frames :],
        )

        # === log results ===
        self.log_dict(
            {f"{namespace}/{k}": v for k, v in metric_dict.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # === clear outputs for next epoch ===
        self.validation_step_outputs.clear()