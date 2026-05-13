"""
Prepare colmap/dense/depths_2/ for dn-splatter / ags-mesh training.

Reads Femto Mega ToF depths from nerfstudio_data/depths_femto/, applies the
confidence mask, resizes to half resolution (matches images_2/), and saves
as float32 .npy in meters. dn-splatter's NormalNerfstudio dataparser looks
up `depths_2/<stem>.npy` when downscale-factor=2.
"""
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "nerfstudio_data" / "depths_femto"
DST = ROOT / "colmap" / "dense" / "depths_2"
DST_FULL = ROOT / "colmap" / "dense" / "depths"
DST.mkdir(parents=True, exist_ok=True)
DST_FULL.mkdir(parents=True, exist_ok=True)

# Match the image resolution used in images_2/ (2x downscale of 1920x1080).
TARGET_W, TARGET_H = 960, 540
# Full resolution (matches images/ folder). colmap_to_ns checks for files at
# this path to populate `depth_file_path` per frame in transforms.json.
FULL_W, FULL_H = 1920, 1080

# Femto valid range (Femto spec: 0.25-5.46 m).
MIN_M, MAX_M = 0.25, 5.3

count, total_pixels, valid_pixels = 0, 0, 0
for depth_path in sorted(SRC.glob("IMG_femto_*.npy")):
    if depth_path.stem.endswith("_conf"):
        continue
    stem = depth_path.stem
    conf_path = SRC / f"{stem}_conf.npy"

    d = np.load(depth_path).astype(np.float32)
    conf = np.load(conf_path).astype(bool) if conf_path.exists() else np.ones_like(d, dtype=bool)

    valid = conf & (d >= MIN_M) & (d <= MAX_M)
    d = np.where(valid, d, 0.0).astype(np.float32)

    # Full-resolution copy (for colmap_to_ns existence check)
    if (d.shape[1], d.shape[0]) != (FULL_W, FULL_H):
        d_full = cv2.resize(d, (FULL_W, FULL_H), interpolation=cv2.INTER_NEAREST)
    else:
        d_full = d
    np.save(DST_FULL / f"{stem}.npy", d_full)

    # 2x-downscaled copy (used at runtime when downscale-factor=2)
    d_half = cv2.resize(d, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)
    np.save(DST / f"{stem}.npy", d_half)

    count += 1
    total_pixels += d_half.size
    valid_pixels += int((d_half > 0).sum())

print(f"Saved {count} depth .npy files to {DST}")
print(f"Resolution: {TARGET_W}x{TARGET_H}")
print(f"Valid pixel rate: {valid_pixels/total_pixels*100:.1f}% (after range + conf mask)")
