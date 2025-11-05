from typing import Optional
import wandb
import numpy as np
import torch

import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import matplotlib.animation as animation
from pathlib import Path
import os
import imageio

plt.set_loglevel("warning")

from torchmetrics.functional import mean_squared_error, peak_signal_noise_ratio
from torchmetrics.functional import (
    structural_similarity_index_measure,
    universal_image_quality_index,
)
from algorithms.common.metrics import (
    FrechetVideoDistance,
    LearnedPerceptualImagePatchSimilarity,
    FrechetInceptionDistance,
)


# FIXME: clean up & check this util
def log_video(
    observation_hat,
    observation_gt=None,
    step=0,
    namespace="train",
    prefix="video",
    context_frames=0,
    color=(255, 0, 0),
    logger=None,
):
    """
    take in video tensors in range [-1, 1] and log into wandb

    :param observation_hat: predicted observation tensor of shape (frame, batch, channel, height, width)
    :param observation_gt: ground-truth observation tensor of shape (frame, batch, channel, height, width)
    :param step: an int indicating the step number
    :param namespace: a string specify a name space this video logging falls under, e.g. train, val
    :param prefix: a string specify a prefix for the video name
    :param context_frames: an int indicating how many frames in observation_hat are ground truth given as context
    :param color: a tuple of 3 numbers specifying the color of the border for ground truth frames
    :param logger: optional logger to use. use global wandb if not specified
    """
    if not logger:
        logger = wandb
    if observation_gt is None:
        observation_gt = torch.zeros_like(observation_hat)
    observation_hat[:context_frames] = observation_gt[:context_frames]
    # Add red border of 1 pixel width to the context frames
    for i, c in enumerate(color):
        c = c / 255.0
        observation_hat[:context_frames, :, i, [0, -1], :] = c
        observation_hat[:context_frames, :, i, :, [0, -1]] = c
        observation_gt[:, :, i, [0, -1], :] = c
        observation_gt[:, :, i, :, [0, -1]] = c
    video = torch.cat([observation_hat, observation_gt], -1).detach().cpu().numpy()
    video = np.transpose(np.clip(video, a_min=0.0, a_max=1.0) * 255, (1, 0, 2, 3, 4)).astype(np.uint8)
    # video[..., 1:] = video[..., :1]  # remove framestack, only visualize current frame
    n_samples = len(video)
    # use wandb directly here since pytorch lightning doesn't support logging videos yet
    for i in range(n_samples):
        logger.log(
            {
                f"{namespace}/{prefix}_{i}": wandb.Video(video[i], fps=24),
                f"trainer/global_step": step,
            }
        )

def get_validation_metrics_for_states(observation_hat, observation_gt):
    """
    :param observation_hat: predicted observation tensor of shape (frame, batch, channel)
    :param observation_gt: ground-truth observation tensor of shape (frame, batch, channel)
    :return: a tuple of metrics
    """
    frame, batch, channel = observation_hat.shape

    # reshape to (frame * batch, channel)
    observation_hat = observation_hat.reshape(-1, channel)
    observation_gt = observation_gt.reshape(-1, channel)

    mse = mean_squared_error(observation_hat, observation_gt)
    psnr = peak_signal_noise_ratio(observation_hat, observation_gt)

    output_dict = {
        "mse": mse,
        "psnr": psnr,
    }

    return output_dict

def log_gaze_video_2d(
    pred,
    gt=None,
    step=0,
    namespace="train",
    prefix="gaze_2d",
    resolution=1408,
    fps=10,
    logger=None,
    dot_radius=6,
    local_downscale=0.5,
    wandb_downscale=0.25,
):
    """
    Efficiently logs 2D gaze trajectories as videos.
    Saves a locally downscaled MP4 (for faster I/O)
    and uploads an even smaller one to WandB.

    Args:
        pred: (T, B, 3, H, W) predicted tensor (normalized by resolution)
        gt: (T, B, 3, H, W) ground-truth tensor (normalized by resolution)
        resolution: normalization factor (e.g., 1408)
        local_downscale: factor for local video size (0.5 → 704×704)
        wandb_downscale: factor for wandb dashboard (0.25 → 352×352)
        fps: frames per second
    """
    if not logger:
        logger = wandb
    if gt is None:
        gt = torch.zeros_like(pred)

    wandb_dir = wandb.run.dir if wandb.run is not None else os.getcwd()
    pred = pred.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()
    T, B = pred.shape[:2]
    res = int(resolution)

    # Precompute gaze points (T, B, 2)
    pred_xy = np.clip(pred[:, :, :2].mean(axis=(-1, -2)) * res, 0, res - 1)
    gt_xy = np.clip(gt[:, :, :2].mean(axis=(-1, -2)) * res, 0, res - 1)

    # Scaled output sizes
    res_local = int(res * local_downscale)
    res_wandb = int(res * wandb_downscale)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7 if res_local >= 512 else 0.4
    thickness = 2 if res_local >= 512 else 1

    for b in range(B):
        local_path = os.path.join(wandb_dir, f"{prefix}_{namespace}_step{step}_b{b}.mp4")
        video_writer = imageio.get_writer(local_path, fps=fps, codec="libx264", quality=7)

        for t in range(T):
            # === Draw on white canvas ===
            canvas = np.ones((res_local, res_local, 3), dtype=np.uint8) * 255

            # Convert normalized positions to downscaled resolution
            px_pred = int(pred_xy[t, b, 0] * local_downscale)
            py_pred = int(pred_xy[t, b, 1] * local_downscale)
            px_gt   = int(gt_xy[t, b, 0] * local_downscale)
            py_gt   = int(gt_xy[t, b, 1] * local_downscale)

            # Draw gaze points
            cv2.circle(canvas, (px_pred, py_pred), dot_radius, (255, 0, 0), -1) # Pred = Red
            cv2.circle(canvas, (px_gt, py_gt), dot_radius, (0, 0, 255), -1)     # GT = Blue

            # Add legend and frame index
            cv2.putText(canvas, "GT = Blue", (20, 40), font, font_scale, (0, 0, 255), thickness)
            cv2.putText(canvas, "Pred = Red", (20, 80), font, font_scale, (255, 0, 0), thickness)
            cv2.putText(canvas, f"t={t:03d}", (res_local - 140, 40), font, font_scale, (0, 0, 0), thickness)

            video_writer.append_data(canvas)

        video_writer.close()

        # === Prepare a smaller version for wandb ===
        # try:
        #     video_small = np.stack([
        #         cv2.resize(cv2.imread(local_path)[..., ::-1], (res_wandb, res_wandb))
        #         for _ in range(1)  # just to ensure shape for wandb
        #     ], axis=0)

        #     logger.log({
        #         f"{namespace}/{prefix}_b{b}": wandb.Video(local_path, fps=fps, format="mp4"),
        #         "trainer/global_step": step,
        #     })
        # except Exception as e:
        #     print(f"[WARN] wandb.Video failed ({e}). Saved only: {local_path}")


def get_validation_metrics_for_videos(
    observation_hat,
    observation_gt,
    lpips_model: Optional[LearnedPerceptualImagePatchSimilarity] = None,
    fid_model: Optional[FrechetInceptionDistance] = None,
    fvd_model: Optional[FrechetVideoDistance] = None,
):
    """
    :param observation_hat: predicted observation tensor of shape (frame, batch, channel, height, width)
    :param observation_gt: ground-truth observation tensor of shape (frame, batch, channel, height, width)
    :param lpips_model: a LearnedPerceptualImagePatchSimilarity object from algorithm.common.metrics
    :param fid_model: a FrechetInceptionDistance object  from algorithm.common.metrics
    :param fvd_model: a FrechetVideoDistance object  from algorithm.common.metrics
    :return: a tuple of metrics
    """
    frame, batch, channel, height, width = observation_hat.shape
    output_dict = {}
    observation_gt = observation_gt.type_as(observation_hat)  # some metrics don't fully support fp16

    if frame < 9:
        fvd_model = None  # FVD requires at least 9 frames

    if fvd_model is not None:
        output_dict["fvd"] = fvd_model.compute(
            torch.clamp(observation_hat, -1.0, 1.0),
            torch.clamp(observation_gt, -1.0, 1.0),
        )

    # reshape to (frame * batch, channel, height, width) for image losses
    observation_hat = observation_hat.view(-1, channel, height, width)
    observation_gt = observation_gt.view(-1, channel, height, width)

    output_dict["mse"] = mean_squared_error(observation_hat, observation_gt)
    output_dict["psnr"] = peak_signal_noise_ratio(observation_hat, observation_gt, data_range=2.0)
    output_dict["ssim"] = structural_similarity_index_measure(observation_hat, observation_gt, data_range=2.0)
    output_dict["uiqi"] = universal_image_quality_index(observation_hat, observation_gt)
    # operations for LPIPS and FID
    observation_hat = torch.clamp(observation_hat, -1.0, 1.0)
    observation_gt = torch.clamp(observation_gt, -1.0, 1.0)

    if lpips_model is not None:
        lpips_model.update(observation_hat, observation_gt)
        lpips = lpips_model.compute().item()
        # Reset the states of non-functional metrics
        output_dict["lpips"] = lpips
        lpips_model.reset()

    if fid_model is not None:
        observation_hat_uint8 = ((observation_hat + 1.0) / 2 * 255).type(torch.uint8)
        observation_gt_uint8 = ((observation_gt + 1.0) / 2 * 255).type(torch.uint8)
        fid_model.update(observation_gt_uint8, real=True)
        fid_model.update(observation_hat_uint8, real=False)
        fid = fid_model.compute()
        output_dict["fid"] = fid
        # Reset the states of non-functional metrics
        fid_model.reset()

    return output_dict

def get_validation_metrics_for_simple(
    observation_hat,
    observation_gt,
):
    """
    :param observation_hat: predicted observation tensor of shape (frame, batch, channel, height, width)
    :param observation_gt: ground-truth observation tensor of shape (frame, batch, channel, height, width)

    :return: a tuple of metrics
    """
    frame, batch, channel, height, width = observation_hat.shape
    output_dict = {}
    observation_gt = observation_gt.type_as(observation_hat)  # some metrics don't fully support fp16

    # reshape to (frame * batch, channel, height, width) for image losses
    observation_hat = observation_hat.view(-1, channel, height, width)
    observation_gt = observation_gt.view(-1, channel, height, width)

    output_dict["mse"] = mean_squared_error(observation_hat, observation_gt)
    output_dict["psnr"] = peak_signal_noise_ratio(observation_hat, observation_gt, data_range=2.0)
    output_dict["ssim"] = structural_similarity_index_measure(observation_hat, observation_gt, data_range=2.0)

    return output_dict
