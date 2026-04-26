"""
Depth map → dense point cloud utilities.

Converts per-frame metric depth maps (from DA3) + camera poses into a
coloured PLY point cloud that Nerfstudio splatfacto can use for
Gaussian initialisation (via transforms.json ply_file_path).
"""

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

logger = logging.getLogger(__name__)


def depth_to_points(
    depth:       np.ndarray,  # [H, W] float32, metres
    conf:        np.ndarray,  # [H, W] float32, 0–1
    color:       np.ndarray,  # [H, W, 3] uint8, RGB
    K:           np.ndarray,  # [3, 3] camera intrinsics
    c2w:         np.ndarray,  # [4, 4] camera-to-world
    stride:      int   = 4,   # subsample every stride-th pixel
    conf_thresh: float = 0.4,
    max_depth_m: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Unproject depth map to 3D world-space points.

    Returns:
        points: [M, 3] float32
        colors: [M, 3] uint8
    """
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Build pixel grid
    us = np.arange(0, W, stride, dtype=np.float32)  # [W/stride]
    vs = np.arange(0, H, stride, dtype=np.float32)  # [H/stride]
    uu, vv = np.meshgrid(us, vs)                    # [H/s, W/s]

    # Sub-sampled maps
    d_sub = depth[::stride, ::stride]
    c_sub = conf[::stride,  ::stride]
    col_sub = color[::stride, ::stride]  # [H/s, W/s, 3]

    # Validity mask
    mask = (
        (d_sub > 0.05) &
        (d_sub < max_depth_m) &
        (c_sub >= conf_thresh)
    )

    z = d_sub[mask]
    u = uu[mask]
    v = vv[mask]

    # Camera-space 3D points
    x_cam = (u - cx) / fx * z
    y_cam = (v - cy) / fy * z
    z_cam = z
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # [M, 3]

    # World-space 3D points
    R_wc = c2w[:3, :3]    # [3, 3]
    t_wc = c2w[:3, 3]     # [3]
    pts_world = (R_wc @ pts_cam.T).T + t_wc  # [M, 3]

    colors = col_sub[mask]  # [M, 3] uint8

    return pts_world.astype(np.float32), colors.astype(np.uint8)


def save_depth_png(
    depth:  np.ndarray,
    path:   Path,
    orig_w: int | None = None,
    orig_h: int | None = None,
) -> None:
    """
    Save metric depth map [H, W] float32 (metres) as 16-bit PNG (millimetres).
    If orig_w/orig_h are given, upsample to that size first (one image at a time,
    so memory stays low even for large original resolutions).
    Load scale factor: 0.001 (mm -> m) in Nerfstudio.
    """
    if orig_w is not None and orig_h is not None and depth.shape != (orig_h, orig_w):
        depth_img = PILImage.fromarray(depth, mode="F")
        depth = np.array(depth_img.resize((orig_w, orig_h), PILImage.BILINEAR), dtype=np.float32)
    depth_mm = (depth * 1000.0).clip(0, 65535).astype(np.uint16)
    PILImage.fromarray(depth_mm).save(path)


def save_ply(points: np.ndarray, colors: np.ndarray, path: Path) -> None:
    """
    Write a coloured point cloud as binary-little-endian PLY.
    points: [N, 3] float32
    colors: [N, 3] uint8
    """
    N = len(points)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    data = np.empty(N, dtype=dtype)
    data["x"] = points[:, 0]
    data["y"] = points[:, 1]
    data["z"] = points[:, 2]
    data["red"]   = colors[:, 0]
    data["green"] = colors[:, 1]
    data["blue"]  = colors[:, 2]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())
    logger.info(f"PLY saved: {N:,} points → {path}")


def build_init_pointcloud(
    depth_maps:       list[np.ndarray],   # [N] × [H, W] float32, metres
    conf_maps:        list[np.ndarray],   # [N] × [H, W] float32
    jpeg_paths:       list[Path],
    intrinsics_list:  list[dict],         # [{fl_x, fl_y, cx, cy}]
    c2w_list:         list[list],         # [N] × [4][4]
    out_ply:          Path,
    stride:           int   = 4,
    conf_thresh:      float = 0.4,
    max_points:       int   = 2_000_000,
) -> Path:
    """
    Merge per-frame depth maps into a single dense PLY point cloud.

    Args:
        stride:      Sub-sample pixels every N steps (4 = 1/16 of pixels).
                     Increase to reduce memory; decrease for denser cloud.
        max_points:  Hard cap on final point count (random downsample if exceeded).
    Returns path to the saved PLY.
    """
    all_pts:    list[np.ndarray] = []
    all_colors: list[np.ndarray] = []

    for i, (depth, conf, jpeg, intr, c2w) in enumerate(
        zip(depth_maps, conf_maps, jpeg_paths, intrinsics_list, c2w_list)
    ):
        img = np.array(PILImage.open(jpeg).convert("RGB"))
        dh, dw = depth.shape  # depth may be at process_res, not full resolution

        # Resize color image to match depth resolution and scale intrinsics
        orig_h, orig_w = img.shape[:2]
        if (orig_h, orig_w) != (dh, dw):
            img = np.array(PILImage.fromarray(img).resize((dw, dh), PILImage.BILINEAR))
            sx, sy = dw / orig_w, dh / orig_h
            K = np.array([[intr["fl_x"] * sx, 0,                intr["cx"] * sx],
                          [0,                 intr["fl_y"] * sy, intr["cy"] * sy],
                          [0,                 0,                 1              ]], dtype=np.float32)
        else:
            K = np.array([[intr["fl_x"], 0,           intr["cx"]],
                          [0,            intr["fl_y"], intr["cy"]],
                          [0,            0,            1         ]], dtype=np.float32)

        c2w_np = np.array(c2w, dtype=np.float32)

        pts, cols = depth_to_points(depth, conf, img, K, c2w_np,
                                    stride=stride, conf_thresh=conf_thresh)
        all_pts.append(pts)
        all_colors.append(cols)
        logger.debug(f"  Frame {i:4d}: {len(pts):,} points")

    all_pts_np    = np.concatenate(all_pts,    axis=0)
    all_colors_np = np.concatenate(all_colors, axis=0)

    # Downsample if too many points
    if len(all_pts_np) > max_points:
        idx = np.random.choice(len(all_pts_np), max_points, replace=False)
        all_pts_np    = all_pts_np[idx]
        all_colors_np = all_colors_np[idx]
        logger.info(f"Downsampled to {max_points:,} points.")

    save_ply(all_pts_np, all_colors_np, out_ply)
    return out_ply
