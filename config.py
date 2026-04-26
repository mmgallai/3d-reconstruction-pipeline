"""
Central configuration for the reconstruction pipeline.
Edit this file to match your machine — do not hardcode paths elsewhere.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# SAM3
# ─────────────────────────────────────────────
SAM3_REPO_PATH   = Path(os.getenv("SAM3_REPO_PATH", r"C:\Users\mgallai\sam3_work\sam3"))
SAM3_CHECKPOINT  = SAM3_REPO_PATH / "checkpoints" / "sam3.pt"

# ─────────────────────────────────────────────
# Docker images
# ─────────────────────────────────────────────
DOCKER_COLMAP       = "colmap/colmap:latest"
DOCKER_OPENMVS      = "yeicor/openmvs-ubuntu-cuda:v2.3.0"
DOCKER_NERFSTUDIO   = "nerfstudio-blackwell"   # built from local Dockerfile
DOCKER_FFMPEG       = "jrottenberg/ffmpeg:6.1-ubuntu"

# ─────────────────────────────────────────────
# COLMAP settings
# ─────────────────────────────────────────────
COLMAP_SINGLE_CAMERA      = 1       # 1 = all images share one camera model (recommended for phone)
COLMAP_CAMERA_MODEL       = "PINHOLE"
COLMAP_MAX_IMAGE_SIZE     = -1      # -1 = full resolution

# ─────────────────────────────────────────────
# Nerfstudio / Gaussian Splatting settings
# ─────────────────────────────────────────────
NERF_DOWNSCALE_FACTOR     = 2       # 1=full, 2=half, 4=quarter. 2 = good balance for rooms.
NERF_MAX_ITERATIONS       = 30000   # 30000 for production, 1000 for quick test
NERF_TRAIN_METHOD         = "splatfacto-big"  # V5: splatfacto-big + DA3 init + scene bounds.
                                              # dn-splatter requires gsplat==1.0.0 — needs separate
                                              # Docker image; tracked as V6 follow-up.

# ─────────────────────────────────────────────
# Mesh export settings
# ─────────────────────────────────────────────
MESH_NUM_FACES            = 300000  # Target face count for Poisson mesh (Quest 3 budget)

# ─────────────────────────────────────────────
# COLMAP MVS settings
# ─────────────────────────────────────────────
COLMAP_MVS_MAX_IMAGE_SIZE = 1600    # Limit MVS image size (px) — controls speed vs quality

# ─────────────────────────────────────────────
# SAM3 masking settings
# ─────────────────────────────────────────────
SAM3_MASK_THRESHOLD       = 0.5     # confidence threshold for mask binarisation
SAM3_MASK_SMOOTH_RADIUS   = 3       # MinFilter radius (pixels) — removes noisy mask edges

# ─────────────────────────────────────────────
# Output settings
# ─────────────────────────────────────────────
EXPORT_MESH               = True    # run OpenMVS textured mesh export
EXPORT_SPLAT              = True    # run Nerfstudio gaussian splat export
