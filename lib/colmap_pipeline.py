"""
COLMAP pipeline steps wrapped as individual functions.
Each step is idempotent — skips if output already exists (checkpoint behaviour).
"""

import logging
from pathlib import Path
from typing import Optional

from lib.docker_runner import run
from config import (DOCKER_COLMAP, COLMAP_SINGLE_CAMERA, COLMAP_CAMERA_MODEL,
                    COLMAP_MAX_IMAGE_SIZE, COLMAP_MVS_MAX_IMAGE_SIZE)

logger = logging.getLogger(__name__)


def _vol(project_root: Path) -> str:
    """Return a Docker volume mount string for the project root."""
    return f'"{project_root}:/project"'


def feature_extraction_and_matching(
    project_root: Path,
    images_rel: str,          # path relative to /project inside container
    db_rel: str,              # path relative to /project inside container
    mask_rel: Optional[str] = None,
) -> None:
    """
    Run COLMAP feature extraction and exhaustive matching.
    mask_rel: if provided, pass as --ImageReader.mask_path (masked SfM).
    """
    mask_arg = f"--ImageReader.mask_path /project/{mask_rel} " if mask_rel else ""

    cmd = (
        f'docker run --rm --gpus all -v {_vol(project_root)} {DOCKER_COLMAP} bash -lc "'
        f'set -e; '
        f'mkdir -p /project/{db_rel.rsplit("/", 1)[0]}; '
        f'colmap feature_extractor '
        f'  --database_path /project/{db_rel} '
        f'  --image_path /project/{images_rel} '
        f'  --ImageReader.single_camera {COLMAP_SINGLE_CAMERA} '
        f'  --ImageReader.camera_model {COLMAP_CAMERA_MODEL} '
        f'  {mask_arg}'
        f'; colmap exhaustive_matcher --database_path /project/{db_rel}'
        f'"'
    )
    run(cmd)


def sparse_reconstruction(
    project_root: Path,
    images_rel: str,
    db_rel: str,
    sparse_out_rel: str,
    refine_intrinsics: bool = True,
) -> None:
    """Run COLMAP mapper to produce a sparse model."""
    ba_flags = "" if refine_intrinsics else (
        "--Mapper.ba_refine_focal_length 0 "
        "--Mapper.ba_refine_principal_point 0 "
        "--Mapper.ba_refine_extra_params 0 "
    )
    cmd = (
        f'docker run --rm --gpus all -v {_vol(project_root)} {DOCKER_COLMAP} bash -lc "'
        f'set -e; '
        f'mkdir -p /project/{sparse_out_rel}; '
        f'colmap mapper '
        f'  --database_path /project/{db_rel} '
        f'  --image_path /project/{images_rel} '
        f'  --output_path /project/{sparse_out_rel} '
        f'  {ba_flags}'
        f'"'
    )
    run(cmd)


def undistort_images(
    project_root: Path,
    images_rel: str,
    sparse_model_rel: str,    # path to the model folder (e.g. colmap/sparse/0)
    dense_out_rel: str,
) -> None:
    """Undistort images using the sparse model. Output is COLMAP format dense folder."""
    max_size = COLMAP_MAX_IMAGE_SIZE
    cmd = (
        f'docker run --rm --gpus all -v {_vol(project_root)} {DOCKER_COLMAP} bash -lc "'
        f'set -e; '
        f'rm -rf /project/{dense_out_rel} && mkdir -p /project/{dense_out_rel}; '
        f'colmap image_undistorter '
        f'  --image_path /project/{images_rel} '
        f'  --input_path /project/{sparse_model_rel} '
        f'  --output_path /project/{dense_out_rel} '
        f'  --output_type COLMAP '
        f'  --max_image_size {max_size}; '
        # Nerfstudio needs sparse/0 subfolder
        f'mkdir -p /project/{dense_out_rel}/sparse/0; '
        f'cp /project/{dense_out_rel}/sparse/*.bin /project/{dense_out_rel}/sparse/0/ 2>/dev/null || true'
        f'"'
    )
    run(cmd)


def best_sparse_model(sparse_dir: Path) -> Path:
    """
    Return the subdirectory (0, 1, 2 ...) that has the largest points3D.bin.
    COLMAP numbers models sequentially — model 0 is NOT always the best one.
    """
    candidates = [d for d in sparse_dir.iterdir() if d.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No sparse model subdirectories found in {sparse_dir}")
    best = max(candidates, key=lambda d: (d / "points3D.bin").stat().st_size
               if (d / "points3D.bin").exists() else 0)
    size_kb = (best / "points3D.bin").stat().st_size / 1024
    logger.info(f"Best sparse model: {best.name}  ({size_kb:.0f} KB in points3D.bin)")
    return best


def run_mvs(
    project_root: Path,
    dense_out_rel: str = "colmap/dense",
    output_ply_rel: str = "colmap/dense/fused.ply",
    max_image_size: int = 1600,
) -> None:
    """
    Run COLMAP Multi-View Stereo: PatchMatchStereo followed by StereoFusion.
    Requires GPU.  Input must be a COLMAP dense folder produced by undistort_images().
    Output: fused dense point cloud at output_ply_rel.
    """
    logger.info("=== COLMAP MVS: PatchMatchStereo + StereoFusion ===")
    cmd = (
        f'docker run --rm --gpus all -v {_vol(project_root)} {DOCKER_COLMAP} bash -lc "'
        f'set -e; '
        f'colmap patch_match_stereo '
        f'  --workspace_path /project/{dense_out_rel} '
        f'  --workspace_format COLMAP '
        f'  --PatchMatchStereo.geom_consistency true '
        f'  --PatchMatchStereo.max_image_size {max_image_size} '
        f'  --PatchMatchStereo.gpu_index -1 '
        f'; colmap stereo_fusion '
        f'  --workspace_path /project/{dense_out_rel} '
        f'  --workspace_format COLMAP '
        f'  --input_type geometric '
        f'  --StereoFusion.min_num_pixels 3 '
        f'  --output_path /project/{output_ply_rel}'
        f'"'
    )
    run(cmd)


def run_full_unmasked_sfm(
    project_root: Path,
    images_rel: str = "colmap/images",
) -> Path:
    """
    Unmasked SfM on raw images → undistorted images in colmap/dense.

    images_rel: path to input images relative to project_root
                (default "colmap/images", can be "nerfstudio_data/images" etc.)

    Returns the relative path (from project_root) to the best sparse model.
    """
    logger.info("=== COLMAP SfM (feature extraction + matching + sparse reconstruction) ===")
    feature_extraction_and_matching(
        project_root,
        images_rel=images_rel,
        db_rel="colmap/database/sfm.db",
    )
    sparse_reconstruction(
        project_root,
        images_rel=images_rel,
        db_rel="colmap/database/sfm.db",
        sparse_out_rel="colmap/sparse",
        refine_intrinsics=True,
    )
    best = best_sparse_model(project_root / "colmap" / "sparse")
    sparse_model_rel = f"colmap/sparse/{best.name}"
    undistort_images(
        project_root,
        images_rel=images_rel,
        sparse_model_rel=sparse_model_rel,
        dense_out_rel="colmap/dense",
    )
    return sparse_model_rel


def run_masked_sfm(project_root: Path) -> None:
    """
    Stage 2: Masked SfM on undistorted images → refined sparse model.
    Masks suppress feature detection on unwanted regions.
    """
    logger.info("=== Stage 2: Masked SfM (on undistorted images) ===")
    feature_extraction_and_matching(
        project_root,
        images_rel="colmap/dense/images",
        db_rel="colmap/database/sfm2_masked.db",
        mask_rel="colmap/dense/masks",
    )
    sparse_reconstruction(
        project_root,
        images_rel="colmap/dense/images",
        db_rel="colmap/database/sfm2_masked.db",
        sparse_out_rel="colmap/dense/sparse_masked",
        refine_intrinsics=False,   # Intrinsics already good from stage 1
    )
