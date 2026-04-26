"""
3D Reconstruction Pipeline
==========================
Turns a folder of photos into:
  - output/splat.ply         Gaussian Splat (best visual quality)
  - output/mesh_textured.ply Textured mesh  (if --mesh flag is set)

Pipeline stages (v5 architecture):
  1. Stage raw images into the workspace
  2. Unmasked SfM   → stable camera poses + undistorted images
  3. SAM3 masking   → binary masks on undistorted images  (optional, --prompt)
  4. Masked SfM     → refined sparse model using masks    (if --prompt)
  5. Safety check   → generate blank masks for any missed images
  6. Nerfstudio     → Gaussian Splat training
  7. Export         → splat.ply + optional mesh
  8. Package        → copy final artifacts to output/

Usage
-----
  python reconstruct.py <images_dir> [options]

Examples
--------
  # Basic (no masking):
  python reconstruct.py assets/sofa_scan

  # With SAM3 masking (remove background, keep sofa):
  python reconstruct.py assets/sofa_scan --prompt "sofa"

  # Also generate textured mesh:
  python reconstruct.py assets/sofa_scan --prompt "sofa" --mesh

  # Quick test run (1000 iterations instead of 30000):
  python reconstruct.py assets/sofa_scan --quick
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

# ─── Project imports ────────────────────────────────────────────────────────
import config
from lib import sam3_helper, colmap_pipeline, nerfstudio_pipeline
from lib.docker_runner import run as docker_run, image_exists

# ─── Logging setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("reconstruct.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def stage_images(images_dir: Path, dest: Path) -> list[Path]:
    """Copy input images to workspace. Returns list of copied files."""
    dest.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in images_dir.iterdir() if f.suffix.lower() in VALID_EXTS)
    if not files:
        raise FileNotFoundError(f"No images found in {images_dir}")
    for f in files:
        shutil.copy2(f, dest / f.name)
    logger.info(f"Staged {len(files)} images to {dest}")
    return files


def run_openmvs_mesh(project_root: Path) -> bool:
    """Run OpenMVS to produce a textured mesh from the COLMAP dense model."""
    P = str(project_root)
    img = config.DOCKER_OPENMVS
    mvs = project_root / "openmvs_mvs"
    mvs.mkdir(exist_ok=True)

    steps = [
        (f'docker run --rm --gpus all -v "{P}/colmap/dense:/data" -v "{P}/openmvs_mvs:/mvs" {img} '
         f'InterfaceCOLMAP -i /data -o /mvs/scene.mvs', False),

        (f'docker run --rm --gpus all -v "{P}/openmvs_mvs:/mvs" {img} '
         f'DensifyPointCloud -i /mvs/scene.mvs', False),

        (f'docker run --rm --gpus all -v "{P}/openmvs_mvs:/mvs" {img} '
         f'ReconstructMesh -i /mvs/scene_dense.mvs', True),

        (f'docker run --rm --gpus all -v "{P}/openmvs_mvs:/mvs" {img} '
         f'TextureMesh -i /mvs/scene_dense.mvs --mesh-file /mvs/scene_mesh.ply', True),
    ]

    for cmd, allow_fail in steps:
        ok = docker_run(cmd, allow_fail=allow_fail)
        if not ok and not allow_fail:
            return False
    return True


def package_outputs(project_root: Path, out_dir: Path) -> None:
    """Copy final artifacts to the clean output directory."""
    out_dir.mkdir(parents=True, exist_ok=True)

    splat_src = project_root / "output" / "splat.ply"
    if splat_src.exists():
        shutil.copy2(splat_src, out_dir / "splat.ply")
        logger.info(f"Splat saved: {out_dir / 'splat.ply'}")

    for tex_name in ["scene_textured.ply", "scene_textured0.png"]:
        src = project_root / "openmvs_mvs" / tex_name
        if src.exists():
            shutil.copy2(src, out_dir / tex_name)
            logger.info(f"Mesh saved: {out_dir / tex_name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="3D Reconstruction Pipeline (Gaussian Splat + optional Mesh)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("images_dir",
                   nargs="?",
                   default=None,
                   help="Path to folder containing input photos. "
                        "Default: reconstruction_project/images/")
    p.add_argument("--project-root", default=None,
                   help="Working directory for all intermediate files. "
                        "Default: reconstruction_project/ next to this script.")
    p.add_argument("--output-dir", default=None,
                   help="Where to put the final outputs. "
                        "Default: <project-root>/output/")
    p.add_argument("--prompt", default=None,
                   help='SAM3 text prompt to mask the target object, e.g. "sofa". '
                        'If omitted, pipeline runs without masking.')
    p.add_argument("--mesh", action="store_true",
                   help="Also generate an OpenMVS textured mesh (slower).")
    p.add_argument("--quick", action="store_true",
                   help="Use 1000 iterations instead of 30000 (for testing).")
    p.add_argument("--skip-training", action="store_true",
                   help="Skip Nerfstudio training and go straight to export "
                        "(useful if training already ran).")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Resolve paths ────────────────────────────────────────────────────────
    script_dir   = Path(__file__).resolve().parent
    project_root = Path(args.project_root).resolve() if args.project_root else script_dir
    images_dir   = Path(args.images_dir).resolve() if args.images_dir else project_root / "images"
    out_dir      = Path(args.output_dir).resolve() if args.output_dir else project_root / "output"

    if not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir}")
        sys.exit(1)

    # ── Override config for quick mode ───────────────────────────────────────
    if args.quick:
        config.NERF_MAX_ITERATIONS = 1000
        config.NERF_DOWNSCALE_FACTOR = 8
        logger.info("Quick mode: 1000 iterations, 8x downscale.")

    logger.info("=" * 60)
    logger.info("  3D Reconstruction Pipeline")
    logger.info(f"  Images:  {images_dir}")
    logger.info(f"  Root:    {project_root}")
    logger.info(f"  Output:  {out_dir}")
    logger.info(f"  Prompt:  {args.prompt or 'None (no masking)'}")
    logger.info(f"  Mesh:    {args.mesh}")
    logger.info("=" * 60)

    # ── 1. Ensure Nerfstudio Docker image ────────────────────────────────────
    nerfstudio_pipeline.ensure_image(script_dir)

    # ── 2. Stage raw images ──────────────────────────────────────────────────
    logger.info("--- Stage 1: Staging images ---")
    colmap_images = project_root / "colmap" / "images"
    stage_images(images_dir, colmap_images)

    # ── 3. Unmasked SfM (always runs — gives us undistorted images) ───────────
    sparse_model_rel = colmap_pipeline.run_full_unmasked_sfm(project_root)

    # ── 4. SAM3 masking (optional) ────────────────────────────────────────────
    masks_dir = project_root / "colmap" / "dense" / "masks"
    undist_dir = project_root / "colmap" / "dense" / "images"
    use_masked_sfm = False

    if args.prompt:
        logger.info(f"--- Stage 3: SAM3 masking (prompt='{args.prompt}') ---")
        try:
            import torch
            device    = "cuda" if torch.cuda.is_available() else "cpu"
            processor = sam3_helper.load_sam3(config.SAM3_REPO_PATH, config.SAM3_CHECKPOINT, device)
            sam3_helper.mask_directory(
                images_dir   = undist_dir,
                masks_dir    = masks_dir,
                prompt       = args.prompt,
                processor    = processor,
                threshold    = config.SAM3_MASK_THRESHOLD,
                smooth_radius= config.SAM3_MASK_SMOOTH_RADIUS,
            )
            use_masked_sfm = True
        except Exception as e:
            logger.error(f"SAM3 masking failed: {e}")
            logger.warning("Continuing without masking.")

    # ── 5. Masked SfM (only if we have masks) ────────────────────────────────
    if use_masked_sfm:
        logger.info("--- Stage 4: Masked SfM ---")
        colmap_pipeline.run_masked_sfm(project_root)
        sparse_model_rel = "colmap/dense/sparse_masked/0"

    # ── 6. Safety: blank masks for any images without one ────────────────────
    if args.prompt:
        logger.info("--- Stage 5: Blank mask safety check ---")
        n = sam3_helper.generate_blank_masks(undist_dir, masks_dir)
        if n > 0:
            logger.info(f"  Generated {n} blank masks for unmasked images.")

    # ── 7. Nerfstudio training ────────────────────────────────────────────────
    if not args.skip_training:
        nerfstudio_pipeline.train(project_root, sparse_model_rel=sparse_model_rel)
    else:
        logger.info("--- Skipping Nerfstudio training (--skip-training) ---")

    # ── 8. Export Gaussian Splat ──────────────────────────────────────────────
    logger.info("--- Stage 7: Exporting Gaussian Splat ---")
    ok = nerfstudio_pipeline.export_splat(project_root)
    if not ok:
        logger.warning("Splat export failed or skipped.")

    # ── 9. Optional: OpenMVS textured mesh ───────────────────────────────────
    if args.mesh:
        logger.info("--- Stage 8: OpenMVS Textured Mesh ---")
        run_openmvs_mesh(project_root)

    # ── 10. Package final outputs ─────────────────────────────────────────────
    logger.info("--- Stage 9: Packaging outputs ---")
    package_outputs(project_root, out_dir)

    logger.info("=" * 60)
    logger.info("  PIPELINE COMPLETE")
    logger.info(f"  Outputs in: {out_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
