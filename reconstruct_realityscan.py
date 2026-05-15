"""
RealityScan → 3D Mesh + Gaussian Splat Pipeline
================================================
Converts a RealityScan export folder (HEIC + XMP sidecars) into:
  - output/mesh.ply      Textured mesh  (for VR physics / Quest 3 interaction)
  - output/splat.ply     Gaussian Splat (for photorealistic VR background)

Pipeline
--------
  1. Convert HEIC → JPEG
  2. COLMAP SfM   — fresh pose estimation from images (no ARKit poses needed)
                    feature extraction + exhaustive matching + sparse reconstruction
                    + image undistortion
  3. COLMAP MVS   — Multi-View Stereo dense reconstruction
                    PatchMatchStereo (GPU) + StereoFusion → fused.ply
  4. transforms.json — convert undistorted COLMAP model to Nerfstudio format
                    fused.ply referenced as ply_file_path (Gaussian init)
  5. 2DGS training — surface-oriented Gaussians (30 000 iters, 2x downscale)
  6. Mesh export  — Poisson surface reconstruction from 2DGS
  7. Splat export — Gaussian Splat .ply for viewing

Quick-start
-----------
  conda activate da3          # or any env with pillow + pillow-heif

  python reconstruct_realityscan.py                   # full quality
  python reconstruct_realityscan.py --quick           # 1000 iters, no MVS (fast test)
  python reconstruct_realityscan.py --skip-mvs        # skip dense MVS, use sparse init
  python reconstruct_realityscan.py --skip-training   # skip training, export only
  python reconstruct_realityscan.py --no-mesh         # skip mesh export
"""

import argparse
import logging
import re
import shutil
import sys
from pathlib import Path

import config
from lib import heic_converter, nerfstudio_pipeline, colmap_pipeline
from lib import colmap_to_ns, openmvs_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("reconstruct_realityscan.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def _next_version(out_dir: Path) -> int:
    """
    Scan out_dir for splat_vN*.ply files and return the next version number.
    e.g. if splat_v1_sparse.ply and splat_v2_mvs.ply exist, returns 3.
    """
    existing = list(out_dir.glob("splat_v*.ply"))
    nums = []
    for f in existing:
        m = re.match(r"splat_v(\d+)", f.name)
        if m:
            nums.append(int(m.group(1)))
    return max(nums, default=0) + 1


def _version_tag(init_ply_rel) -> str:
    """Return a short descriptive tag for the actual init used."""
    if init_ply_rel == "hybrid_init.ply":
        return "hybrid"
    elif init_ply_rel == "da3_init.ply":
        return "da3"
    elif init_ply_rel == "fused.ply":
        return "mvs_full"
    elif init_ply_rel == "fused_sub.ply":
        return "mvs_500k"
    elif init_ply_rel == "sparse_init.ply":
        return "sparse"
    return "noinit"


def parse_args():
    p = argparse.ArgumentParser(
        description="RealityScan → Mesh + Gaussian Splat  (COLMAP + 2DGS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "scan_dir", nargs="?", default=None,
        help="Folder containing RealityScan output (HEIC files). "
             "Default: <project-root>/images/",
    )
    p.add_argument("--project-root", default=None,
                   help="Working directory for intermediate files. Default: script folder.")
    p.add_argument("--output-dir", default=None,
                   help="Destination for final outputs. Default: <project-root>/output/")
    p.add_argument("--jpeg-quality", type=int, default=92)

    # -- Mode flags -----------------------------------------------------------
    p.add_argument("--quick", action="store_true",
                   help="1000 iterations, 4x downscale, skip MVS (fast test).")
    p.add_argument("--downscale-factor", type=int, default=None,
                   help="Override NERF_DOWNSCALE_FACTOR (config.py). "
                        "1=full res, 2=half (default), 4=quarter. Bigger GPU "
                        "→ smaller factor → sharper output but more VRAM.")
    p.add_argument("--iters", type=int, default=None,
                   help="Override NERF_MAX_ITERATIONS (config.py). "
                        "Default 30000; raise to 50000+ on stronger hardware "
                        "for marginally higher splat fidelity.")
    p.add_argument("--skip-mvs", action="store_true",
                   help="Skip COLMAP MVS — use sparse COLMAP points as Gaussian init.")
    p.add_argument("--skip-training", action="store_true",
                   help="Skip 2DGS training — jump straight to export.")
    p.add_argument("--resume", action="store_true",
                   help="Resume the most recent interrupted training run from its last checkpoint.")
    p.add_argument("--no-mesh", action="store_true",
                   help="Skip nerfstudio Poisson mesh export (splat only).")
    p.add_argument("--no-splat", action="store_true",
                   help="Skip Gaussian splat export (mesh only).")
    p.add_argument("--train-method", type=str, default=None,
                   help="Override NERF_TRAIN_METHOD from config. Examples: "
                        "splatfacto, splatfacto-big (default), dn-splatter, "
                        "dn-splatter-big, ags-mesh. V22 used ags-mesh — first apply "
                        "patches/dn_splatter_for_gsplat_15.patch to your dn-splatter/ clone.")
    p.add_argument("--use-femto-depth", action="store_true",
                   help="V23: Use Orbbec Femto Mega ToF depth (from "
                        "nerfstudio_data/depths_femto/) instead of running DA3 "
                        "monocular depth. Requires `capture_femto.py` run first. "
                        "Builds tof_init.ply from back-projected depths + COLMAP poses.")
    # ── OpenMVS textured-mesh path (parallel to 3DGS) ────────────────────────
    p.add_argument("--openmvs-mesh", action="store_true", default=True,
                   help="Run OpenMVS textured mesh in parallel to 3DGS (default ON).")
    p.add_argument("--no-openmvs-mesh", dest="openmvs_mesh", action="store_false",
                   help="Disable the OpenMVS textured-mesh path.")
    p.add_argument("--refine-mesh", action="store_true", default=True,
                   help="Enable RefineMesh — photometric refinement (~10 min). "
                        "Part of the canonical V17 recipe — ON by default.")
    p.add_argument("--no-refine-mesh", dest="refine_mesh", action="store_false",
                   help="Disable RefineMesh photometric refinement (faster, "
                        "lower geometric fidelity).")
    p.add_argument("--no-lod", action="store_true",
                   help="Skip LOD decimation (mid 500K / low 100K). Default: produce LODs.")
    p.add_argument("--no-ao", action="store_true",
                   help="Skip per-vertex ambient-occlusion bake. Default: bake AO on each LOD.")
    # Experimental: Poisson reconstruction. Tested at depth 12 in V19/V20 — produced
    # noisier surfaces and lower visual quality than the default Delaunay path despite
    # higher polygon count. Kept for further experimentation but NOT recommended.
    p.add_argument("--poisson-mesh", action="store_true",
                   help="EXPERIMENTAL: use Open3D screened Poisson reconstruction "
                        "instead of OpenMVS Delaunay graph-cut. V20 testing showed "
                        "this produces noisier, hole-prone surfaces — prefer the default.")
    p.add_argument("--poisson-depth", type=int, default=11,
                   help="Poisson octree depth: 10≈1-3M verts, 11≈3-8M, 12≈8-20M (default 11).")

    # -- MVS quality ----------------------------------------------------------
    p.add_argument("--mvs-max-image-size", type=int, default=None,
                   help="Limit image size for PatchMatchStereo (default: from config). "
                        "Reduce to 800-1200 if MVS runs out of GPU memory.")
    return p.parse_args()


def _banner(project_root, scan_dir, out_dir, args):
    logger.info("=" * 60)
    logger.info("  RealityScan → Mesh + Gaussian Splat Pipeline")
    logger.info(f"  Scan:         {scan_dir}")
    logger.info(f"  Root:         {project_root}")
    logger.info(f"  Output:       {out_dir}")
    logger.info(f"  Method:       COLMAP SfM + {'MVS' if not args.skip_mvs else 'sparse'} → 2DGS")
    logger.info(f"  Iterations:   {config.NERF_MAX_ITERATIONS}")
    logger.info(f"  Downscale:    {config.NERF_DOWNSCALE_FACTOR}x")
    logger.info("=" * 60)


def main():
    args = parse_args()

    script_dir   = Path(__file__).resolve().parent
    project_root = Path(args.project_root).resolve() if args.project_root else script_dir
    scan_dir     = Path(args.scan_dir).resolve() if args.scan_dir else project_root / "images"
    out_dir      = Path(args.output_dir).resolve() if args.output_dir else project_root / "output"

    if not scan_dir.exists():
        logger.error(f"Scan directory not found: {scan_dir}")
        sys.exit(1)

    if args.train_method:
        config.NERF_TRAIN_METHOD = args.train_method
        logger.info(f"Train method overridden: NERF_TRAIN_METHOD = {args.train_method}")

    if args.quick:
        config.NERF_MAX_ITERATIONS   = 1000
        config.NERF_DOWNSCALE_FACTOR = 4
        args.skip_mvs = True
        logger.info("Quick mode: 1000 iterations, 4x downscale, MVS skipped.")

    # Explicit CLI overrides take precedence over config.py defaults but lose
    # to --quick (above), since quick is the explicit "I want a fast test" knob.
    if args.iters is not None:
        config.NERF_MAX_ITERATIONS = args.iters
        logger.info(f"Iterations overridden: NERF_MAX_ITERATIONS = {args.iters}")
    if args.downscale_factor is not None:
        config.NERF_DOWNSCALE_FACTOR = args.downscale_factor
        logger.info(f"Downscale overridden: NERF_DOWNSCALE_FACTOR = {args.downscale_factor}")

    _banner(project_root, scan_dir, out_dir, args)

    # Paths used throughout
    jpeg_dir    = project_root / "nerfstudio_data" / "images"
    colmap_dir  = project_root / "colmap"
    dense_dir   = colmap_dir / "dense"
    fused_ply   = dense_dir / "fused.ply"
    sparse_ply  = dense_dir / "sparse_init.ply"
    ns_json     = dense_dir / "transforms.json"

    # ── 1. Ensure Nerfstudio Docker image ─────────────────────────────────────
    if not args.skip_training:
        nerfstudio_pipeline.ensure_image(script_dir)

    # ── 2. HEIC → JPEG ────────────────────────────────────────────────────────
    # Femto Mega captures direct to JPEG via capture_femto.py, so we skip this
    # stage when --use-femto-depth is on. The images are already in jpeg_dir.
    if args.use_femto_depth:
        existing_jpegs = list(jpeg_dir.glob("*.jpg")) + list(jpeg_dir.glob("*.jpeg"))
        if not existing_jpegs:
            logger.error(f"--use-femto-depth set but no JPEGs in {jpeg_dir}.\n"
                         f"  Run `python capture_femto.py` first.")
            sys.exit(1)
        logger.info(f"--- Stage 1: HEIC→JPEG (skipping — using {len(existing_jpegs)} Femto JPEGs) ---")
    else:
        logger.info("--- Stage 1: Converting HEIC → JPEG ---")
        converted = heic_converter.convert_directory(
            scan_dir, jpeg_dir, quality=args.jpeg_quality,
        )
        if not converted:
            logger.error("No JPEG images produced. Aborting.")
            sys.exit(1)
        logger.info(f"  {len(converted)} images in {jpeg_dir}")

    # ── 3. COLMAP SfM ─────────────────────────────────────────────────────────
    undistorted_images_dir = dense_dir / "images"
    undistorted_sparse_dir = dense_dir / "sparse" / "0"

    sfm_done = (
        undistorted_sparse_dir.exists()
        and (undistorted_sparse_dir / "cameras.bin").exists()
        and undistorted_images_dir.exists()
        and any(undistorted_images_dir.iterdir())
    )

    if sfm_done:
        logger.info("--- Stage 2: COLMAP SfM (skipping — undistorted model already exists) ---")
    else:
        logger.info("--- Stage 2: COLMAP SfM (feature extraction → sparse model → undistortion) ---")
        # Preserve fused.ply before undistort_images wipes colmap/dense
        fused_backup = colmap_dir / "fused_backup.ply"
        if fused_ply.exists() and not fused_backup.exists():
            shutil.copy2(fused_ply, fused_backup)
            logger.info(f"  Preserved fused.ply → {fused_backup}")

        images_rel = jpeg_dir.relative_to(project_root).as_posix()

        # V32: if Femto intrinsics JSON is present, seed COLMAP with the
        # vendor PINHOLE params and lock them during bundle adjustment.
        # SfM becomes faster and more robust (no joint focal/principal
        # solve), and the back-projection in femto_to_init.py uses the
        # measured camera, not COLMAP's estimate.
        intr_json = project_root / "nerfstudio_data" / "femto_intrinsics.json"
        cam_model_arg = None
        cam_params_arg = None
        if args.use_femto_depth and intr_json.exists():
            import json as _json
            try:
                intr = _json.loads(intr_json.read_text())["color_intrinsics"]
                cam_model_arg  = "PINHOLE"
                cam_params_arg = (f"{intr['fx']:.4f},{intr['fy']:.4f},"
                                  f"{intr['cx']:.4f},{intr['cy']:.4f}")
                logger.info(f"  Femto vendor intrinsics found → {cam_model_arg} "
                            f"{cam_params_arg} (locked during BA)")
            except Exception as _e:
                logger.warning(f"  Could not read {intr_json.name}: {_e}; "
                               f"falling back to COLMAP-estimated intrinsics")

        colmap_pipeline.run_full_unmasked_sfm(
            project_root, images_rel=images_rel,
            camera_model=cam_model_arg,
            camera_params=cam_params_arg,
        )

        # Restore fused.ply if it was backed up
        if fused_backup.exists() and not fused_ply.exists():
            shutil.copy2(fused_backup, fused_ply)
            logger.info(f"  Restored fused.ply from backup.")

    if not undistorted_sparse_dir.exists():
        logger.error(f"COLMAP undistortion output not found: {undistorted_sparse_dir}")
        sys.exit(1)

    # ── 4. COLMAP MVS ─────────────────────────────────────────────────────────
    init_ply_rel = None   # relative to ns_json (transforms.json) directory
    skip_mvs     = args.skip_mvs

    if not skip_mvs:
        logger.info("--- Stage 3: COLMAP MVS (PatchMatchStereo + StereoFusion) ---")
        max_img_size = (
            args.mvs_max_image_size
            if args.mvs_max_image_size
            else config.COLMAP_MVS_MAX_IMAGE_SIZE
        )
        try:
            colmap_pipeline.run_mvs(
                project_root,
                dense_out_rel="colmap/dense",
                output_ply_rel="colmap/dense/fused.ply",
                max_image_size=max_img_size,
            )
            if fused_ply.exists():
                size_mb = fused_ply.stat().st_size / 1_048_576
                logger.info(f"  MVS dense point cloud: {size_mb:.1f} MB → {fused_ply}")
                init_ply_rel = "fused.ply"   # relative to dense_dir (transforms.json location)
            else:
                logger.warning("fused.ply not produced — falling back to sparse COLMAP init.")
                skip_mvs = True
        except Exception as exc:
            logger.warning(f"MVS failed ({exc}) — falling back to sparse COLMAP init.")
            skip_mvs = True

    if skip_mvs:
        # If fused.ply exists, subsample it to keep GPU memory under control,
        # then use the subsampled version as Gaussian init.
        fused_sub_ply = dense_dir / "fused_sub.ply"
        if fused_ply.exists():
            size_mb = fused_ply.stat().st_size / 1_048_576
            logger.info(f"--- Stage 3: MVS skipped — subsampling fused.ply ({size_mb:.1f} MB) → fused_sub.ply ---")
            import subprocess as _sp
            _sp.run([
                "conda", "run", "-n", "da3", "python",
                str(project_root / "subsample_ply.py"),
                "--input",      str(fused_ply),
                "--output",     str(fused_sub_ply),
                "--target",     "500000",
                "--sigma-clip", "3.0",
            ], check=True)
            init_ply_rel = "fused_sub.ply"
        else:
            # No fused.ply — fall back to sparse COLMAP points
            logger.info("--- Stage 3 (fallback): Exporting sparse COLMAP point cloud ---")
            n = colmap_to_ns.export_sparse_ply(undistorted_sparse_dir, sparse_ply)
            if n > 0:
                init_ply_rel = "sparse_init.ply"
            else:
                logger.warning("No sparse points exported — 2DGS will use random init.")

    # ── 5. Depth map generation ───────────────────────────────────────────────
    # Two sources: DA3 monocular (default), or Femto Mega ToF (--use-femto-depth).
    # Femto gives TRUE metric depth — no scale alignment needed.
    depth_dir    = dense_dir / "depths"
    da3_init_ply = dense_dir / "da3_init.ply"
    da3_bounds_f = dense_dir / "da3_bounds.json"
    tof_init_ply = dense_dir / "tof_init.ply"
    tof_bounds_f = dense_dir / "tof_bounds.json"
    da3_bounds   = None

    if args.use_femto_depth:
        # ── ToF init: back-project Femto depths + COLMAP poses → tof_init.ply ──
        femto_depths = project_root / "nerfstudio_data" / "depths_femto"
        if not femto_depths.exists() or not any(femto_depths.glob("*.npy")):
            logger.error(f"--use-femto-depth set but no .npy files under {femto_depths}.\n"
                         f"  Run `python capture_femto.py` first.")
            sys.exit(1)
        if tof_init_ply.exists():
            logger.info("--- Stage 4: Femto ToF init (skipping — tof_init.ply already exists) ---")
        else:
            logger.info("--- Stage 4: Femto ToF init (back-project depths + COLMAP poses) ---")
            import subprocess as _sp2
            _sp2.run([
                "conda", "run", "-n", "da3", "python",
                str(project_root / "femto_to_init.py"),
                "--project-root", str(project_root),
                "--target",       "500000",
                "--sigma-clip",   "3.0",
                "--max-depth",    "5.3",
                "--min-depth",    "0.30",
            ], check=True)
        if tof_bounds_f.exists():
            import json as _json
            with open(tof_bounds_f) as _f:
                da3_bounds = _json.load(_f)
            logger.info(f"  ToF scene bounds loaded from {tof_bounds_f}")
    else:
        # ── DA3 monocular path (original V6 / V14 recipe) ──
        depth_maps_done = (
            depth_dir.exists()
            and len(list(depth_dir.glob("*.npy"))) >= n_frames if 'n_frames' in dir() else False
            or (depth_dir.exists() and any(depth_dir.glob("*.npy")))
        )

        if depth_maps_done and da3_init_ply.exists():
            logger.info("--- Stage 4: DA3 depth generation (skipping — already exists) ---")
        else:
            logger.info("--- Stage 4: DA3 depth generation (Depth Anything 3 Metric-Large) ---")
            import subprocess as _sp2
            _sp2.run([
                "conda", "run", "-n", "da3", "python",
                str(project_root / "generate_da3_depths.py"),
                "--project-root", str(project_root),
                "--target",       "500000",
                "--sigma-clip",   "3.0",
                "--max-depth",    "12.0",
                "--batch-size",   "32",
                "--model",        "da3metric-large",
            ], check=True)

        if da3_bounds_f.exists():
            import json as _json
            with open(da3_bounds_f) as _f:
                da3_bounds = _json.load(_f)
            logger.info(f"  DA3 scene bounds loaded from {da3_bounds_f}")

    # ── 4b. Hybrid init (MVS + scale-aligned DA3) ─────────────────────────────
    # Skipped when --use-femto-depth: ToF init is already metric and dense,
    # no need to fuse with MVS.
    hybrid_init_ply = dense_dir / "hybrid_init.ply"
    fused_sub_ply   = dense_dir / "fused_sub.ply"

    # Build hybrid_init.ply if both source clouds exist and the hybrid is older
    # than either source (or missing entirely).
    if (not args.use_femto_depth) and fused_sub_ply.exists() and da3_init_ply.exists():
        need_rebuild = (
            not hybrid_init_ply.exists()
            or hybrid_init_ply.stat().st_mtime < fused_sub_ply.stat().st_mtime
            or hybrid_init_ply.stat().st_mtime < da3_init_ply.stat().st_mtime
        )
        if need_rebuild:
            logger.info("--- Stage 4b: Building hybrid init (MVS + scale-aligned DA3) ---")
            import subprocess as _sp3
            _sp3.run([
                "conda", "run", "-n", "da3", "python",
                str(project_root / "generate_hybrid_init.py"),
                "--mvs-ply",  str(fused_sub_ply),
                "--da3-ply",  str(da3_init_ply),
                "--output",   str(hybrid_init_ply),
                "--target",   "750000",
            ], check=True)
        else:
            logger.info("--- Stage 4b: Hybrid init (skipping — up to date) ---")

    # Init priority: ToF (Femto) > hybrid > DA3 > MVS_500k > sparse
    if args.use_femto_depth and tof_init_ply.exists():
        size_mb = tof_init_ply.stat().st_size / 1_048_576
        logger.info(f"  Using ToF init cloud (Femto Mega): {size_mb:.1f} MB → {tof_init_ply.name}")
        init_ply_rel = "tof_init.ply"
    elif hybrid_init_ply.exists():
        size_mb = hybrid_init_ply.stat().st_size / 1_048_576
        logger.info(f"  Using HYBRID init cloud: {size_mb:.1f} MB → {hybrid_init_ply.name}")
        init_ply_rel = "hybrid_init.ply"
    elif da3_init_ply.exists():
        size_mb = da3_init_ply.stat().st_size / 1_048_576
        logger.info(f"  Using DA3 init cloud: {size_mb:.1f} MB → {da3_init_ply.name}")
        init_ply_rel = "da3_init.ply"

    # ── 6. Generate transforms.json ───────────────────────────────────────────
    logger.info("--- Stage 5: Converting COLMAP model → transforms.json ---")
    n_frames = colmap_to_ns.convert(
        sparse_model_dir = undistorted_sparse_dir,
        images_dir       = undistorted_images_dir,
        output_json      = ns_json,
        ply_file_path    = init_ply_rel,
        depths_dir       = depth_dir if depth_dir.exists() else None,
        da3_bounds       = da3_bounds,
    )
    if n_frames == 0:
        logger.error("transforms.json has 0 frames. Aborting.")
        sys.exit(1)
    logger.info(f"  {n_frames} frames → {ns_json}")

    # ── 7. Pre-generate downscaled images ─────────────────────────────────────
    logger.info(f"--- Stage 6: Pre-generating images_{config.NERF_DOWNSCALE_FACTOR}/ ---")
    nerfstudio_pipeline.pregenerate_downscaled_images(
        dense_dir, config.NERF_DOWNSCALE_FACTOR,
    )

    # ── 8. Training ───────────────────────────────────────────────────────────
    data_rel = dense_dir.relative_to(project_root).as_posix()
    if args.skip_training:
        logger.info("--- Skipping training (--skip-training) ---")
    elif args.resume:
        logger.info("--- Stage 7: Resuming interrupted training ---")
        nerfstudio_pipeline.resume_training(project_root, data_rel=data_rel)
    else:
        logger.info(f"--- Stage 7: Training ({config.NERF_TRAIN_METHOD}, "
                    f"init={init_ply_rel or 'random'}) ---")
        nerfstudio_pipeline.train_nerfstudio_format(
            project_root, data_rel=data_rel,
        )

    # ── 9. Export mesh (nerfstudio Poisson — usually fails on 3DGS) ───────────
    if not args.no_mesh:
        logger.info("--- Stage 8: Exporting mesh from 3DGS (Poisson) ---")
        ok = nerfstudio_pipeline.export_mesh(project_root, method="poisson")
        if not ok:
            logger.warning("3DGS Poisson mesh export failed (expected — 3DGS lacks "
                           "surface normals). Use OpenMVS path instead (Stage 8b).")

    # ── 9b. OpenMVS textured-mesh path (parallel to 3DGS) ─────────────────────
    openmvs_mesh_path = None
    if args.openmvs_mesh:
        # Skip if already done for this dense workspace
        existing_mesh = project_root / "openmvs" / "scene_textured.ply"
        fused_ply     = dense_dir / "fused.ply"
        if (existing_mesh.exists()
                and fused_ply.exists()
                and existing_mesh.stat().st_mtime > fused_ply.stat().st_mtime):
            logger.info("--- Stage 8b: OpenMVS textured mesh (skipping — already up to date) ---")
            openmvs_mesh_path = existing_mesh
        else:
            logger.info("--- Stage 8b: OpenMVS textured-mesh pipeline ---")
            try:
                openmvs_mesh_path = openmvs_pipeline.run_full_mesh_pipeline(
                    project_root,
                    colmap_dense_rel = "colmap/dense",
                    pointcloud_rel   = None,  # use depth maps in scene.mvs (avoids SIGSEGV)
                    do_refine        = args.refine_mesh,
                    use_poisson      = args.poisson_mesh,
                    poisson_depth    = args.poisson_depth,
                )
            except Exception as exc:
                logger.warning(f"OpenMVS mesh pipeline raised: {exc}")
                openmvs_mesh_path = None

    # ── 10. Export splat ──────────────────────────────────────────────────────
    if not args.no_splat:
        logger.info("--- Stage 9: Exporting Gaussian Splat ---")
        ok = nerfstudio_pipeline.export_splat(project_root)
        if not ok:
            logger.warning("Splat export failed or skipped.")

    # ── 11. Package outputs ───────────────────────────────────────────────────
    logger.info("--- Stage 10: Packaging outputs ---")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine version number and init tag
    version = _next_version(out_dir)
    tag     = _version_tag(init_ply_rel)
    label   = f"v{version}_{tag}"
    logger.info(f"  Version label: {label}")

    # Map raw nerfstudio output names → versioned output names
    artifacts = {
        "splat.ply": f"splat_{label}.ply",
        "mesh.ply":  f"mesh_{label}.ply",
        "mesh.obj":  f"mesh_{label}.obj",
    }
    for raw_name, versioned_name in artifacts.items():
        src = project_root / "output" / raw_name
        dst = out_dir / versioned_name
        if src.exists():
            shutil.copy2(src, dst)
            size_mb = dst.stat().st_size / 1_048_576
            logger.info(f"  {raw_name} → {versioned_name}  ({size_mb:.1f} MB)")

    # Copy versioned OpenMVS textured mesh (if produced this run).
    # Mesh + its textures go into a per-version subdir so MeshLab can resolve
    # the "comment TextureFile scene_textured*.png" entries in the .ply
    # without renaming. Each version is self-contained and viewable in place.
    if openmvs_mesh_path is not None and openmvs_mesh_path.exists():
        mesh_subdir = out_dir / f"mesh_v{version}"
        mesh_subdir.mkdir(parents=True, exist_ok=True)
        omvs_dst = mesh_subdir / f"mesh_v{version}_openmvs.ply"
        shutil.copy2(openmvs_mesh_path, omvs_dst)
        size_mb = omvs_dst.stat().st_size / 1_048_576
        logger.info(f"  OpenMVS mesh → {mesh_subdir.name}/{omvs_dst.name}  ({size_mb:.1f} MB)")
        # Pick up .obj+.mtl if TextureMesh produced UV-mapped output
        for ext in (".obj", ".mtl"):
            src = openmvs_mesh_path.with_suffix(ext)
            if src.exists():
                shutil.copy2(src, mesh_subdir / f"mesh_v{version}_openmvs{ext}")
        # Textures keep their original names (referenced by the .ply header)
        for tex in openmvs_mesh_path.parent.glob("scene_textured*.png"):
            shutil.copy2(tex, mesh_subdir / tex.name)

        # Tier-3 polish: LOD decimation (mid/low) + per-vertex AO bake on HIGH
        if not args.no_lod:
            import subprocess as _sp
            logger.info("--- Stage 10b: LOD decimation (mid 500K / low 100K) ---")
            _sp.run(["conda", "run", "-n", "da3", "--no-capture-output",
                     "python", str(project_root / "lib" / "mesh_lod.py"),
                     str(omvs_dst), str(mesh_subdir),
                     "--mid", "500000", "--low", "100000"], check=False)
        if not args.no_ao:
            import subprocess as _sp
            logger.info("--- Stage 10c: Ambient-occlusion bake (HIGH/MID/LOW) ---")
            for src, rays in [
                (omvs_dst, 32),
                (mesh_subdir / f"mesh_v{version}_openmvs_mid.ply", 64),
                (mesh_subdir / f"mesh_v{version}_openmvs_low.ply", 64),
            ]:
                if src.exists():
                    dst = src.with_name(src.stem + "_ao.ply")
                    _sp.run(["conda", "run", "-n", "da3", "--no-capture-output",
                             "python", str(project_root / "lib" / "mesh_ao_bake.py"),
                             str(src), str(dst),
                             "--rays", str(rays), "--max-dist", "0.5"], check=False)

    # Auto-run prune on the splat and save versioned pruned copy.
    # Three-stage filter: opacity + σ-outlier + connected-component.
    raw_splat = project_root / "output" / "splat.ply"
    if raw_splat.exists():
        pruned_name = f"splat_{label}_pruned.ply"
        pruned_dst  = out_dir / pruned_name
        logger.info(f"  Running prune → {pruned_name} ...")
        import subprocess
        subprocess.run(
            ["conda", "run", "-n", "da3", "python",
             str(project_root / "prune_splat.py"),
             "--input",         str(raw_splat),
             "--output",        str(pruned_dst),
             "--threshold",     "0.15",
             "--sigma-outlier", "5.0",
             "--cc-voxel",      "0.10",
             "--cc-min-frac",   "0.005"],
            check=False,
        )
        if pruned_dst.exists():
            size_mb = pruned_dst.stat().st_size / 1_048_576
            logger.info(f"  Pruned splat saved: {pruned_name}  ({size_mb:.1f} MB)")

    logger.info("=" * 60)
    logger.info("  PIPELINE COMPLETE")
    logger.info(f"  Output dir : {out_dir}")
    logger.info(f"  Version    : {label}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
