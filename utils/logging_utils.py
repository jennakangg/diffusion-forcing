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
import csv
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

def log_real_video_with_gaze(
    pred,
    gt,
    raw_gaze,
    video_paths,
    start_idxs,
    logger,
    namespace="validation_real",
    step=0,
    resolution=1408,
    frame_skip=1,
    local_downscale=1,
    fps=30,
    dot_radius=20,
):
    """
    Real-video overlay logger matching the SAME API as log_gaze_video_2d:

        pred: (T, B, 3, H, W)
        gt:   (T, B, 3, H, W)
        video_paths: list/tuple of length B
        start_idxs:  list/tuple of length B

    Creates one MP4 per batch element with gaze overlaid on real RGB frames.
    """
    if not logger:
        logger = wandb

    wandb_dir = logger.experiment.dir if hasattr(logger, "experiment") else (
        wandb.run.dir if wandb.run is not None else os.getcwd()
    )

    pred_np = pred.detach().cpu().numpy()
    gt_np   = gt.detach().cpu().numpy()

    if len(raw_gaze) == 1:
        raw_gaze = raw_gaze[0]
    raw_np = raw_gaze.detach().cpu().numpy()  # (T, B, 2)
    raw_np = raw_np.transpose(1, 0, 2)

   
    T, B = gt_np.shape[:2]
    res = int(resolution)

    # Precompute gaze points exactly like log_gaze_video_2d
    pred_xy = np.clip(pred_np[:, :, :2].mean(axis=(-1, -2)) * res, 0, res - 1)   # (T, B, 2)
    gt_xy   = np.clip(gt_np[:, :, :2].mean(axis=(-1, -2)) * res, 0, res - 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    if len(video_paths) == 1 and isinstance(video_paths[0], (list, tuple)):
        video_paths = video_paths[0]

    if len(start_idxs) == 1 and torch.is_tensor(start_idxs[0]):
        start_idxs = start_idxs[0]


    # === Loop over batch elements ===
    for b in range(B):

        # --- Extract correct video path ---
        vid_path = video_paths[b]
        if isinstance(vid_path, (list, tuple)):
            vid_path = vid_path[0]
        if isinstance(vid_path, torch.Tensor):
            vid_path = vid_path[0].item() if vid_path.numel() > 1 else vid_path.item()
        vid_path = str(vid_path)

        # --- Extract start idx ---
        s_idx = start_idxs[b]
        if isinstance(s_idx, (list, tuple)):
            s_idx = s_idx[0]
        if isinstance(s_idx, torch.Tensor):
            s_idx = s_idx[0].item() if s_idx.numel() > 1 else s_idx.item()
        start_idx = int(s_idx)

        # --- Load raw RGB frames ---
        raw_frames = load_raw_frames(
            vid_path,
            start_idx=start_idx,
            T=T,
            frame_skip=frame_skip,
        )   # (T, H, W, 3)

        H, W = res, res
        out_h, out_w = int(H * local_downscale), int(W * local_downscale)

        # Writer
        save_path = os.path.join(
            wandb_dir,
            f"realgaze_{namespace}_step{step}_b{b}.mp4"
        )
        writer = imageio.get_writer(save_path, fps=fps, codec="libx264", quality=7)

        # === Write frames ===
        for t in range(T):
            frame = (raw_frames[t] * 255).astype(np.uint8)
            frame = cv2.resize(frame, (out_w, out_h))

            # Convert normalized coords
            px_pred = int(pred_xy[t, b, 0] * (out_w / res))
            py_pred = int(pred_xy[t, b, 1] * (out_h / res))

            px_gt   = int(gt_xy[t, b, 0]   * (out_w / res))
            py_gt   = int(gt_xy[t, b, 1]   * (out_h / res))
   
            # raw_gt_xy gives TRUE pixel coords
            raw_x = int(raw_np[t, b, 0])
            raw_y = int(raw_np[t, b, 1])

            cv2.circle(frame, (raw_x, raw_y), dot_radius, (0, 255, 0), -1)

            # Draw
            cv2.circle(frame, (px_pred, py_pred), dot_radius, (255, 0, 0), -1)
            cv2.circle(frame, (px_gt,   py_gt),   dot_radius, (0, 0, 255), -1)

            # === Text overlays with white background ===
            # 1. Legend
            legend = "GT = Blue / Pred = Red / Raw = Green"
            (tx_w, tx_h), base = cv2.getTextSize(legend, font, font_scale, thickness)
            x, y = 20, 40
            cv2.rectangle(frame,
                        (x, y - tx_h - base),
                        (x + tx_w + 6, y + 6),
                        (255, 255, 255),
                        -1)
            cv2.putText(frame, legend, (x + 3, y),
                        font, font_scale, (0, 0, 0), thickness)
            parts = vid_path.split("\\")
            short = "\\".join(parts[-3::2])  # take folder just above frame_aligned_videos + filename

            # 2. Filename + start idx
            info = f"{short}  start={start_idx}"
            (tx_w, tx_h), base = cv2.getTextSize(info, font, font_scale, thickness)
            x, y = 20, 80
            cv2.rectangle(frame,
                        (x, y - tx_h - base),
                        (x + tx_w + 6, y + 6),
                        (255, 255, 255),
                        -1)
            cv2.putText(frame, info, (x + 3, y),
                        font, font_scale, (0, 0, 0), thickness)

            # 3. Frame index
            t_text = f"t={t:03d}"
            (tx_w, tx_h), base = cv2.getTextSize(t_text, font, font_scale, thickness)
            x = out_w - tx_w - 20
            y = 40
            cv2.rectangle(frame,
                        (x, y - tx_h - base),
                        (x + tx_w + 6, y + 6),
                        (255, 255, 255),
                        -1)
            cv2.putText(frame, t_text, (x + 3, y),
                        font, font_scale, (0, 0, 0), thickness)


            writer.append_data(frame)

        writer.close()


def load_raw_frames(video_path: str, start_idx: int, T: int, frame_skip: int):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

    frames = []
    for _ in range(T):
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb)

        if frame_skip > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES,
                    cap.get(cv2.CAP_PROP_POS_FRAMES) + (frame_skip - 1))

    cap.release()
    return np.stack(frames) / 255.0   # (T, H, W, 3)

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
