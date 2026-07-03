"""
Build the splat-init point cloud from Femto Mega depth maps + COLMAP poses.

Equivalent to `generate_da3_depths.py` but skips DA3 inference entirely —
Femto already gives metric depth. Result drops straight into the pipeline
as `colmap/dense/tof_init.ply`.

Note: even though Femto depths are metric, COLMAP poses are arbitrary-scale,
so we still solve a COLMAP↔metric scale factor against the sparse 3D points
(same approach as `generate_da3_depths.py`) before back-projecting.

Usage:
  conda run -n da3 python femto_to_init.py
  python femto_to_init.py --target 500000 --sigma-clip 3.0 --max-depth 5.3

Inputs:
  nerfstudio_data/depths_femto/IMG_femto_NNNN.npy        (per-frame metric depth, meters)
  nerfstudio_data/depths_femto/IMG_femto_NNNN_conf.npy   (per-frame valid mask, uint8)
  nerfstudio_data/images/IMG_femto_NNNN.jpg              (per-frame RGB)
  colmap/sparse/0/{cameras,images,points3D}.bin          (COLMAP poses + intrinsics)

Output:
  colmap/dense/tof_init.ply                              (point cloud, ~500K verts)
  colmap/dense/tof_bounds.json                           (scene AABB metadata)
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

try:
    import cv2  # only for image read; downscale RGB to depth resolution
except ImportError:
    cv2 = None

# Reuse the COLMAP readers + utilities from generate_da3_depths.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_da3_depths as g_da3  # noqa


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", default=None,
                   help="Pipeline root (default: script folder)")
    p.add_argument("--depths-dir", default="nerfstudio_data/depths_femto",
                   help="Folder with per-frame Femto depth/conf .npy files")
    p.add_argument("--images-dir", default="nerfstudio_data/images",
                   help="Folder with the captured RGB JPEGs (for vertex colours)")
    p.add_argument("--colmap-sparse", default="colmap/dense/sparse/0",
                   help="COLMAP sparse model folder. Default is the UNDISTORTED "
                        "model in dense/ — that is the 'best' model COLMAP picked "
                        "(may have all frames) after image_undistorter ran. The raw "
                        "colmap/sparse/0/ often contains only a small sub-cluster.")
    p.add_argument("--output-ply", default="colmap/dense/tof_init.ply",
                   help="Destination init.ply (relative to project-root)")
    p.add_argument("--bounds-json", default="colmap/dense/tof_bounds.json",
                   help="Destination scene-bounds JSON")
    p.add_argument("--target", type=int, default=500_000,
                   help="Target point count after voxel subsample")
    p.add_argument("--sigma-clip", type=float, default=3.0,
                   help="Reject points beyond N·σ from the centroid (default 3.0)")
    p.add_argument("--max-depth", type=float, default=5.3,
                   help="Drop depths beyond this (Femto spec: 5.46 m, default 5.3)")
    p.add_argument("--min-depth", type=float, default=0.30,
                   help="Drop depths closer than this (Femto spec: 0.25 m)")
    # ── Hybrid ToF + DA3 fallback (fills <0.25 m and >5.3 m gaps) ────────────
    p.add_argument("--use-da3-fallback", action="store_true",
                   help="Fill out-of-ToF-range pixels with DA3 depths, per-frame "
                        "linear-fit (α·d_da3 + β) anchored to ToF on overlap pixels.")
    p.add_argument("--da3-depths-dir", default="colmap/dense/depths",
                   help="Folder with DA3 .npy depth maps (from generate_da3_depths.py).")
    p.add_argument("--da3-max-depth", type=float, default=12.0,
                   help="Cap DA3-fill depths at this many metres (default 12).")
    p.add_argument("--min-overlap", type=int, default=200,
                   help="Min ToF/DA3 overlap pixels per frame to fit α,β. "
                        "Below this, fall back to global DA3 scale (default 200).")
    return p.parse_args()


def main():
    args = _parse_args()
    root = Path(args.project_root or __file__).resolve()
    if root.is_file():
        root = root.parent

    depths_dir = root / args.depths_dir
    images_dir = root / args.images_dir
    sparse_dir = root / args.colmap_sparse
    out_ply    = root / args.output_ply
    bounds_js  = root / args.bounds_json

    print(f"=== Femto → init.ply ===")
    print(f"  depths:    {depths_dir}")
    print(f"  images:    {images_dir}")
    print(f"  COLMAP:    {sparse_dir}")
    print(f"  output:    {out_ply}")

    if not (depths_dir / next(depths_dir.glob("*.npy"), Path("__none__")).name).exists():
        print(f"  ERROR: no depth .npy files in {depths_dir}", file=sys.stderr)
        return 2

    # ── Load COLMAP poses + intrinsics + sparse points (for scale solve) ────
    cameras  = g_da3._read_cameras(sparse_dir / "cameras.bin")
    images   = g_da3._read_images(sparse_dir / "images.bin", with_points2d=True)
    points3d = g_da3._read_points3d_xyz(sparse_dir / "points3D.bin")
    print(f"  COLMAP: {len(cameras)} camera(s), {len(images)} registered image(s), "
          f"{len(points3d):,} sparse 3D points")

    # ── Solve COLMAP→metric scale ───────────────────────────────────────────
    # COLMAP poses are arbitrary-scale; Femto depths are metric. Back-projecting
    # metric depth through arbitrary-scale poses produces nonsense. Use the
    # sparse 3D points to solve scale = z_colmap / z_femto (median), then
    # multiply every Femto depth by this scale before back-projection.
    def _femto_depth_loader(img_meta):
        stem = Path(img_meta["name"]).stem
        f = depths_dir / f"{stem}.npy"
        if not f.exists():
            return None
        d = np.load(f).astype(np.float32)
        cam = cameras[img_meta["camera_id"]]
        cam_h, cam_w = cam["height"], cam["width"]
        if d.shape != (cam_h, cam_w):
            if cv2 is None:
                return None
            d = cv2.resize(d, (cam_w, cam_h), interpolation=cv2.INTER_NEAREST)
        # Zero out out-of-range depths so they don't poison the ratio
        d = np.where((d >= args.min_depth) & (d <= args.max_depth), d, 0.0)
        return d

    print("  solving COLMAP↔Femto scale factor ...")
    scale_factor, scale_info = g_da3.compute_global_scale(
        images, cameras, points3d, _femto_depth_loader,
    )
    print(f"    pairs: {scale_info['n_pairs']:,} from {scale_info['n_images']} images")
    print(f"    ratio (colmap/femto): median={scale_info['median']:.4f}  "
          f"p25={scale_info['p25']:.4f}  p75={scale_info['p75']:.4f}  "
          f"std={scale_info['std']:.4f}")
    print(f"    → multiplying every Femto depth by {scale_factor:.4f} (COLMAP-scale)")

    # ── Optional: solve DA3 scale + prepare per-frame fit ──────────────────
    da3_depths_dir = root / args.da3_depths_dir
    da3_scale = None
    da3_info  = None
    if args.use_da3_fallback:
        if not da3_depths_dir.exists() or not any(da3_depths_dir.glob("*.npy")):
            print(f"  ERROR: --use-da3-fallback set but no .npy in {da3_depths_dir}",
                  file=sys.stderr)
            return 5

        def _da3_depth_loader(img_meta):
            stem = Path(img_meta["name"]).stem
            f = da3_depths_dir / f"{stem}.npy"
            if not f.exists():
                return None
            d = np.load(f).astype(np.float32)
            cam = cameras[img_meta["camera_id"]]
            cam_h, cam_w = cam["height"], cam["width"]
            if d.shape != (cam_h, cam_w):
                if cv2 is None:
                    return None
                d = cv2.resize(d, (cam_w, cam_h), interpolation=cv2.INTER_NEAREST)
            return d

        print("  solving COLMAP↔DA3 scale factor (for global fallback) ...")
        da3_scale, da3_info = g_da3.compute_global_scale(
            images, cameras, points3d, _da3_depth_loader,
        )
        print(f"    pairs: {da3_info['n_pairs']:,} from {da3_info['n_images']} images")
        print(f"    ratio (colmap/da3): median={da3_info['median']:.4f}  "
              f"p25={da3_info['p25']:.4f}  p75={da3_info['p75']:.4f}")
        # DA3-to-Femto-metric scale: if no overlap pixels in a frame, this maps
        # DA3 metres → Femto-anchored metres so it goes through scale_factor cleanly.
        da3_to_femto_metric = da3_scale / scale_factor
        print(f"    global DA3→Femto-metric factor = {da3_to_femto_metric:.4f} "
              f"({(da3_to_femto_metric-1)*100:+.1f}% vs ToF)")

    # ── Per-frame back-projection ──────────────────────────────────────────
    all_xyz = []
    all_rgb = []
    rng = np.random.default_rng(0)

    # Per-frame fit diagnostics (only if DA3 fallback enabled)
    fit_alphas, fit_betas, fit_overlaps, fit_fallbacks = [], [], [], 0

    # _read_images returns a list of dicts; _read_cameras returns a dict keyed by camera_id.
    for img in images:
        # Match COLMAP image name → our depth/conf .npy
        img_name = img["name"]
        stem = Path(img_name).stem
        depth_path = depths_dir / f"{stem}.npy"
        conf_path  = depths_dir / f"{stem}_conf.npy"
        rgb_path   = images_dir / img_name

        if not depth_path.exists():
            # COLMAP also keeps non-Femto images? Skip silently.
            continue
        if not rgb_path.exists():
            print(f"  WARN: missing RGB for {img_name}, skipping")
            continue

        depth = np.load(depth_path).astype(np.float32)
        conf  = np.load(conf_path).astype(bool) if conf_path.exists() \
                else np.ones_like(depth, dtype=bool)

        # COLMAP intrinsics — produced at the RGB resolution we fed it
        cam = cameras[img["camera_id"]]
        K = g_da3._intrinsics_from_camera(cam)
        w2c = g_da3._w2c_from_image(img)

        # If depth was D2C-aligned to color, it's already at color resolution.
        # Verify shape matches the COLMAP camera dims; if not, resize.
        cam_h, cam_w = cam["height"], cam["width"]
        if depth.shape != (cam_h, cam_w):
            if cv2 is None:
                print(f"  ERROR: depth {depth.shape} != camera {(cam_h, cam_w)} "
                      f"and opencv not installed for resize", file=sys.stderr)
                return 3
            depth = cv2.resize(depth, (cam_w, cam_h), interpolation=cv2.INTER_NEAREST)
            conf  = cv2.resize(conf.astype(np.uint8), (cam_w, cam_h),
                              interpolation=cv2.INTER_NEAREST).astype(bool)

        # ToF range mask (in Femto-metric, BEFORE scale-up to COLMAP units)
        tof_valid = conf & (depth >= args.min_depth) & (depth <= args.max_depth)

        # SOTA: Prune retroreflective monitor IR ghosts. Active IR ToF sensors 
        # suffer from multipath mirror reflections on shiny glass screens, returning 
        # saturated/high intensity returns (e.g. >60,000 in 16-bit).
        ir_path = depths_dir / f"{stem}_ir.png"
        if ir_path.exists() and cv2 is not None:
            ir_img = cv2.imread(str(ir_path), cv2.IMREAD_UNCHANGED)
            if ir_img is not None:
                if ir_img.shape != depth.shape:
                    ir_img = cv2.resize(ir_img, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
                tof_valid = tof_valid & (ir_img < 60000)

        # ── Hybrid: build a per-pixel depth map combining ToF + DA3 fallback
        if args.use_da3_fallback:
            da3_path = da3_depths_dir / f"{stem}.npy"
            if da3_path.exists():
                da3 = np.load(da3_path).astype(np.float32)
                if da3.shape != (cam_h, cam_w):
                    da3 = cv2.resize(da3, (cam_w, cam_h),
                                     interpolation=cv2.INTER_NEAREST)

                # Per-frame fit: anchor DA3 to ToF using their overlap pixels
                overlap = tof_valid & (da3 > 0.05)
                if int(overlap.sum()) >= args.min_overlap:
                    A = np.stack([da3[overlap], np.ones(overlap.sum())], axis=1)
                    sol, *_ = np.linalg.lstsq(A, depth[overlap], rcond=None)
                    alpha, beta = float(sol[0]), float(sol[1])
                    da3_corrected = alpha * da3 + beta
                    fit_alphas.append(alpha); fit_betas.append(beta)
                    fit_overlaps.append(int(overlap.sum()))
                else:
                    # Not enough overlap — use global DA3→Femto-metric scale
                    da3_corrected = da3 * da3_to_femto_metric
                    fit_fallbacks += 1

                # DA3 fills only where ToF is invalid AND the corrected DA3
                # is sane (positive, within da3_max_depth).
                da3_fill = (~tof_valid) & (da3_corrected > args.min_depth * 0.5) \
                           & (da3_corrected <= args.da3_max_depth)

                # Final per-pixel depth (Femto-metric)
                depth = np.where(tof_valid, depth, 0.0)
                depth = np.where(da3_fill, da3_corrected, depth)
                valid = tof_valid | da3_fill
            else:
                valid = tof_valid
        else:
            valid = tof_valid

        if not valid.any():
            continue

        # Back-project valid pixels — scale metric depth → COLMAP scale so the
        # back-projected points land in the same coord frame as the poses.
        vs, us = np.where(valid)
        ds     = depth[vs, us] * scale_factor
        # Camera-space: X = (u-cx)/fx * d, Y = (v-cy)/fy * d, Z = d
        cx, cy = K[0, 2], K[1, 2]
        fx, fy = K[0, 0], K[1, 1]
        x_cam = (us - cx) / fx * ds
        y_cam = (vs - cy) / fy * ds
        z_cam = ds
        pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)  # N×3

        # Camera-to-world: pts_world = R^T (pts_cam - t)
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        pts_world = (pts_cam - t) @ R  # equiv (R^T @ (pts - t).T).T

        # Per-pixel RGB
        bgr = cv2.imread(str(rgb_path)) if cv2 else None
        if bgr is not None:
            if bgr.shape[:2] != (cam_h, cam_w):
                bgr = cv2.resize(bgr, (cam_w, cam_h), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)[vs, us]
        else:
            rgb = np.full((pts_world.shape[0], 3), 180, dtype=np.uint8)

        # Per-image cap: don't let one image dominate
        cap = 20_000
        if pts_world.shape[0] > cap:
            idx = rng.choice(pts_world.shape[0], cap, replace=False)
            pts_world = pts_world[idx]
            rgb       = rgb[idx]

        all_xyz.append(pts_world)
        all_rgb.append(rgb)

    if not all_xyz:
        print("  ERROR: no valid frames produced any points", file=sys.stderr)
        return 4

    xyz = np.concatenate(all_xyz, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)
    print(f"  fused: {len(xyz):,} world-space points from {len(all_xyz)} frame(s)")

    if args.use_da3_fallback and fit_alphas:
        a = np.array(fit_alphas); b = np.array(fit_betas)
        ov = np.array(fit_overlaps)
        print(f"  per-frame DA3→ToF fit  (n={len(a)} frames, "
              f"{fit_fallbacks} global-scale fallback):")
        print(f"    α  median={np.median(a):.3f}  p25={np.percentile(a,25):.3f}  "
              f"p75={np.percentile(a,75):.3f}")
        print(f"    β  median={np.median(b):.3f}  p25={np.percentile(b,25):.3f}  "
              f"p75={np.percentile(b,75):.3f}  (metres)")
        print(f"    overlap median={int(np.median(ov)):,} px / frame")

    # ── σ-clip outliers ────────────────────────────────────────────────────
    if args.sigma_clip > 0:
        c = xyz.mean(axis=0)
        d = np.linalg.norm(xyz - c, axis=1)
        thresh = d.mean() + args.sigma_clip * d.std()
        keep = d <= thresh
        before = len(xyz)
        xyz = xyz[keep]
        rgb = rgb[keep]
        print(f"  σ-clipped @ {args.sigma_clip}σ: {before:,} → {len(xyz):,}")

    # ── Voxel subsample to target count ────────────────────────────────────
    if args.target and len(xyz) > args.target:
        bb = xyz.max(0) - xyz.min(0)
        # Coarse iterative voxel size search
        vox = max(bb.max() / 200, 1e-3)
        for _ in range(20):
            xyz_sub, rgb_sub = g_da3._voxel_subsample(xyz, rgb, vox)
            if len(xyz_sub) <= args.target * 1.05:
                break
            vox *= 1.15
        xyz, rgb = xyz_sub, rgb_sub
        print(f"  voxel-subsampled to {len(xyz):,} points (voxel={vox*1000:.1f} mm)")

    # ── Write PLY ──────────────────────────────────────────────────────────
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    ply_bytes = g_da3._build_ply(xyz.astype(np.float32), rgb.astype(np.uint8))
    out_ply.write_bytes(ply_bytes)
    print(f"  saved: {out_ply}  ({out_ply.stat().st_size/1024/1024:.1f} MB)")

    # ── Write bounds JSON ──────────────────────────────────────────────────
    bb_min = xyz.min(0).tolist()
    bb_max = xyz.max(0).tolist()
    bounds_js.parent.mkdir(parents=True, exist_ok=True)
    bounds_dict = {
        "x_min": bb_min[0], "x_max": bb_max[0],
        "y_min": bb_min[1], "y_max": bb_max[1],
        "z_min": bb_min[2], "z_max": bb_max[2],
        "scale_factor_da3_to_colmap": float(scale_factor),  # name kept for pipeline compat
        "scale_aligned": True,
        "scale_info": scale_info,
        "source": ("femto_mega_tof+da3_fill" if args.use_da3_fallback else "femto_mega_tof"),
        "n_points": int(len(xyz)),
    }
    if args.use_da3_fallback:
        bounds_dict["da3_scale_factor"] = float(da3_scale)
        bounds_dict["da3_scale_info"]   = da3_info
        if fit_alphas:
            bounds_dict["fit_summary"] = {
                "n_frames_fit": len(fit_alphas),
                "n_frames_global_fallback": int(fit_fallbacks),
                "alpha_median": float(np.median(fit_alphas)),
                "beta_median":  float(np.median(fit_betas)),
                "overlap_median_px": int(np.median(fit_overlaps)) if fit_overlaps else 0,
            }
    bounds_js.write_text(json.dumps(bounds_dict, indent=2))
    print(f"  saved: {bounds_js}")
    print(f"\n=== Done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
