"""
Stage 4: Depth Anything 3 depth map generation + point cloud fusion.

Reads the COLMAP undistorted model, runs DA3Metric-Large on every image
with COLMAP camera poses supplied so the output is metric and aligned to
the COLMAP coordinate system.  Produces:

  colmap/dense/depths/<image>.npy   — per-frame float32 depth maps (metres)
                                      at original image resolution
  colmap/dense/da3_init.ply         — dense RGB point cloud (scene-bounded,
                                      500 k pts default) for Gaussian init
  colmap/dense/da3_bounds.json      — scene AABB used to constrain training

Usage (called from reconstruct_realityscan.py, or standalone):
    conda run -n da3 python generate_da3_depths.py \\
        --project-root  <path>   \\
        --target        500000   \\
        --sigma-clip    3.0      \\
        --batch-size    32
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage


# ── helpers identical to colmap_to_ns.py ─────────────────────────────────────

_MODEL_NUM_PARAMS = {
    0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 12, 6: 9, 7: 5, 8: 4, 9: 5, 10: 13,
}


def _read_cameras(path):
    cameras = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            (cid,)   = struct.unpack("<I", f.read(4))
            (mid,)   = struct.unpack("<i", f.read(4))
            (w,)     = struct.unpack("<Q", f.read(8))
            (h,)     = struct.unpack("<Q", f.read(8))
            np_      = _MODEL_NUM_PARAMS.get(mid, 4)
            params   = struct.unpack(f"<{np_}d", f.read(8 * np_))
            cameras[cid] = dict(model_id=mid, width=int(w), height=int(h), params=params)
    return cameras


def _read_images(path, with_points2d=False):
    """
    Read COLMAP images.bin.
    If with_points2d=True, also reads the per-image (x, y, point3D_id) arrays
    needed for scale alignment with DA3.
    """
    images = []
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            (iid,)  = struct.unpack("<I", f.read(4))
            qvec    = struct.unpack("<4d", f.read(32))
            tvec    = struct.unpack("<3d", f.read(24))
            (cid,)  = struct.unpack("<I", f.read(4))
            name_b  = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_b += c
            name = name_b.decode("utf-8", errors="replace")
            (np2d,) = struct.unpack("<Q", f.read(8))
            entry = dict(image_id=iid, qvec=qvec, tvec=tvec,
                         camera_id=cid, name=name)
            if with_points2d and np2d > 0:
                # Each 2D point: x (double), y (double), point3D_id (int64; -1 = no track)
                raw = f.read(24 * np2d)
                arr = np.frombuffer(raw, dtype=[
                    ("x",  "<f8"), ("y",  "<f8"), ("p3d", "<i8")
                ])
                # keep only points with a valid 3D track
                mask = arr["p3d"] >= 0
                entry["points2D"] = np.stack(
                    [arr["x"][mask], arr["y"][mask]], axis=1
                ).astype(np.float64)
                entry["points3D_ids"] = arr["p3d"][mask].astype(np.int64)
            else:
                f.read(24 * np2d)
                entry["points2D"]    = np.zeros((0, 2), dtype=np.float64)
                entry["points3D_ids"] = np.zeros((0,), dtype=np.int64)
            images.append(entry)
    return images


def _read_points3d_xyz(path):
    """Return {point3D_id: np.array([x,y,z])} from points3D.bin."""
    pts = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            (pid,)  = struct.unpack("<Q", f.read(8))
            xyz     = struct.unpack("<3d", f.read(24))
            f.read(3)                          # rgb
            f.read(8)                          # reprojection error
            (ntrk,) = struct.unpack("<Q", f.read(8))
            f.read(8 * ntrk)                   # track elements (image_id + point2D_idx)
            pts[pid] = np.array(xyz, dtype=np.float64)
    return pts


def _qvec_to_rotmat(qvec):
    qw, qx, qy, qz = qvec
    return np.array([
        [1-2*(qy**2+qz**2),  2*(qx*qy-qz*qw),  2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),  1-2*(qx**2+qz**2),  2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),  2*(qy*qz+qx*qw),  1-2*(qx**2+qy**2)],
    ], dtype=np.float64)


def _intrinsics_from_camera(cam):
    p, mid = cam["params"], cam["model_id"]
    if mid == 0:
        return np.array([[p[0], 0, p[1]], [0, p[0], p[2]], [0, 0, 1]])
    elif mid in (1, 2, 3, 4):
        return np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1]])
    else:
        return np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1]])


def _w2c_from_image(img):
    """COLMAP image → 4×4 world-to-camera matrix."""
    R = _qvec_to_rotmat(img["qvec"])
    t = np.array(img["tvec"], dtype=np.float64)
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3,  3] = t
    return w2c


# ── point-cloud helpers ───────────────────────────────────────────────────────

def compute_global_scale(images, cameras, points3d, depth_loader,
                         max_per_image=2000, min_total_pairs=200):
    """
    Compute the global median scale factor that converts DA3 metric depth
    into COLMAP world-scale depth.

    Returns
    -------
    scale : float   # multiply DA3 depth maps by this to get COLMAP-scale depth
    info  : dict    # {'n_pairs', 'n_images', 'p25', 'median', 'p75', 'std'}

    Logic (per the expert's recipe):
      For every image, project each visible COLMAP-3D-point through the
      camera, take its z-component (= COLMAP depth), and look up the DA3
      depth at the same pixel.  Pool ratios across all images, take the
      median (robust to COLMAP outliers and DA3 edge errors).
    """
    all_ratios = []
    n_imgs_used = 0

    for img_meta in images:
        if len(img_meta["points2D"]) == 0:
            continue

        depth_orig = depth_loader(img_meta)
        if depth_orig is None:
            continue
        H, W = depth_orig.shape

        cam = cameras[img_meta["camera_id"]]
        K   = _intrinsics_from_camera(cam)
        w2c = _w2c_from_image(img_meta)

        pix2d = img_meta["points2D"]
        p3ids = img_meta["points3D_ids"]

        # Look up the COLMAP-3D coordinates of each point
        valid_mask = np.array([pid in points3d for pid in p3ids], dtype=bool)
        if valid_mask.sum() < 5:
            continue

        pix2d = pix2d[valid_mask]
        xyz   = np.stack([points3d[p] for p in p3ids[valid_mask]], axis=0)  # (N,3)

        # Project into camera (COLMAP w2c is OpenCV: p_cam = R @ p_w + t)
        ones    = np.ones((len(xyz), 1))
        homog   = np.hstack([xyz, ones])              # (N,4)
        p_cam   = (w2c @ homog.T).T[:, :3]            # (N,3)
        z_colmap = p_cam[:, 2]                        # COLMAP-scale z-depth
        # discard points behind the camera or absurdly far
        in_front = (z_colmap > 0.01) & (z_colmap < 100.0)
        if in_front.sum() < 5:
            continue
        pix2d    = pix2d[in_front]
        z_colmap = z_colmap[in_front]

        # Sample DA3 depth at the same pixel coords (nearest neighbour for safety)
        u = np.clip(np.round(pix2d[:, 0]).astype(np.int32), 0, W - 1)
        v = np.clip(np.round(pix2d[:, 1]).astype(np.int32), 0, H - 1)
        z_da3 = depth_orig[v, u]                      # in DA3 metres

        valid_d = (z_da3 > 0.05) & (z_da3 < 30.0)
        if valid_d.sum() < 5:
            continue

        ratios_i = z_colmap[valid_d] / z_da3[valid_d]

        # Per-image cap to prevent any single frame from dominating
        if len(ratios_i) > max_per_image:
            idx = np.random.default_rng(0).choice(len(ratios_i), max_per_image, replace=False)
            ratios_i = ratios_i[idx]

        all_ratios.append(ratios_i)
        n_imgs_used += 1

    if not all_ratios:
        raise RuntimeError("No valid (COLMAP, DA3) depth pairs found — "
                           "cannot compute scale factor.")

    ratios = np.concatenate(all_ratios)
    if len(ratios) < min_total_pairs:
        raise RuntimeError(f"Only {len(ratios)} (COLMAP, DA3) pairs found; "
                           f"need at least {min_total_pairs} for a robust median.")

    info = dict(
        n_pairs  = int(len(ratios)),
        n_images = int(n_imgs_used),
        p25      = float(np.percentile(ratios, 25)),
        median   = float(np.median(ratios)),
        p75      = float(np.percentile(ratios, 75)),
        std      = float(np.std(ratios)),
    )
    return info["median"], info


def _backproject_frame(depth_m, conf, K, w2c, conf_pct=50, max_depth=15.0):
    """
    Back-project a single depth map to world-space XYZ.
    Returns (N,3) float32 array.
    conf may be None (da3metric-large does not produce confidence maps).
    """
    H, W = depth_m.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    valid = (depth_m > 0.05) & (depth_m < max_depth)
    if conf is not None and conf.size > 0:
        conf_thresh = np.percentile(conf, conf_pct)
        valid &= (conf >= conf_thresh)

    v, u = np.where(valid)
    d     = depth_m[v, u].astype(np.float64)

    # Camera-space points
    x_c = (u - cx) / fx * d
    y_c = (v - cy) / fy * d
    z_c = d
    pts_cam = np.stack([x_c, y_c, z_c, np.ones_like(z_c)], axis=1)  # (N,4)

    # World-space: apply c2w = inv(w2c)
    c2w = np.linalg.inv(w2c)
    pts_world = (c2w @ pts_cam.T).T[:, :3]
    return pts_world.astype(np.float32)


def _build_ply(xyz, rgb=None):
    """Build a binary-little-endian PLY bytes object."""
    n = len(xyz)
    has_rgb = rgb is not None
    prop_lines = (
        "property float x\nproperty float y\nproperty float z\n"
        + ("property uchar red\nproperty uchar green\nproperty uchar blue\n" if has_rgb else "")
    )
    header = (
        f"ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        f"{prop_lines}"
        f"end_header\n"
    ).encode("ascii")

    if has_rgb:
        data = np.zeros(n, dtype=[("x","f4"),("y","f4"),("z","f4"),
                                   ("r","u1"),("g","u1"),("b","u1")])
        data["x"] = xyz[:, 0]; data["y"] = xyz[:, 1]; data["z"] = xyz[:, 2]
        data["r"] = rgb[:, 0]; data["g"] = rgb[:, 1]; data["b"] = rgb[:, 2]
    else:
        data = np.zeros(n, dtype=[("x","f4"),("y","f4"),("z","f4")])
        data["x"] = xyz[:, 0]; data["y"] = xyz[:, 1]; data["z"] = xyz[:, 2]

    return header + data.tobytes()


def _voxel_subsample(xyz, rgb, voxel_size):
    """Reduce point cloud density uniformly via voxel grid (one point per voxel)."""
    if voxel_size <= 0:
        return xyz, rgb
    keys = np.floor(xyz / voxel_size).astype(np.int32)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return xyz[idx], rgb[idx] if rgb is not None else None


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", default=None,
                   help="Pipeline root (default: script folder).")
    p.add_argument("--target",      type=int,   default=500_000,
                   help="Target point count for da3_init.ply (default: 500 000).")
    p.add_argument("--sigma-clip",  type=float, default=3.0,
                   help="Spatial outlier σ threshold (0=off).")
    p.add_argument("--max-depth",   type=float, default=12.0,
                   help="Maximum valid depth in metres (default: 12).")
    p.add_argument("--conf-pct",    type=float, default=40.0,
                   help="Confidence percentile cutoff per frame (default: 40).")
    p.add_argument("--batch-size",  type=int,   default=32,
                   help="Images per DA3 inference batch (default: 32).")
    p.add_argument("--model",       default="da3metric-large",
                   help="DA3 model name (default: da3metric-large).")
    p.add_argument("--fuse-only",   action="store_true",
                   help="Skip DA3 inference — re-fuse existing depth maps into PLY only.")
    p.add_argument("--no-rescale",  action="store_true",
                   help="Skip COLMAP↔DA3 scale alignment (NOT recommended; produces a "
                        "scale-mismatched init cloud).")
    return p.parse_args()


def main():
    args   = parse_args()
    root   = Path(args.project_root).resolve() if args.project_root \
             else Path(__file__).resolve().parent
    sparse = root / "colmap" / "dense" / "sparse" / "0"
    img_dir = root / "colmap" / "dense" / "images"
    depth_dir = root / "colmap" / "dense" / "depths"
    out_ply   = root / "colmap" / "dense" / "da3_init.ply"
    out_bounds = root / "colmap" / "dense" / "da3_bounds.json"

    depth_dir.mkdir(parents=True, exist_ok=True)

    # ── Load COLMAP model ──────────────────────────────────────────────────────
    print(f"Reading COLMAP model from {sparse} ...")
    cameras = _read_cameras(sparse / "cameras.bin")
    # Need points2D + points3D for scale alignment
    images   = _read_images(sparse / "images.bin", with_points2d=True)
    points3d = _read_points3d_xyz(sparse / "points3D.bin")
    images   = sorted(images, key=lambda x: x["name"])
    print(f"  {len(cameras)} camera(s), {len(images)} images, {len(points3d):,} 3D points")

    # Filter to images that exist on disk
    images = [img for img in images if (img_dir / img["name"]).exists()]
    print(f"  {len(images)} images found on disk")
    if not images:
        print("ERROR: No images found. Aborting.")
        sys.exit(1)

    # ── Compute COLMAP↔DA3 global scale factor ────────────────────────────────
    # (only done in fuse-only mode where depth maps already exist; for the
    # primary inference path we compute it after all depths have been written)
    scale_factor = 1.0
    scale_info   = None

    def _load_existing_depth(img_meta):
        f = depth_dir / f"{Path(img_meta['name']).stem}.npy"
        return np.load(f) if f.exists() else None

    if args.fuse_only and not args.no_rescale:
        print("\nComputing global COLMAP↔DA3 scale factor ...")
        scale_factor, scale_info = compute_global_scale(
            images, cameras, points3d, _load_existing_depth,
        )
        print(f"  Pairs: {scale_info['n_pairs']:,}  from {scale_info['n_images']} images")
        print(f"  ratio (colmap/da3)  median={scale_info['median']:.4f}  "
              f"p25={scale_info['p25']:.4f}  p75={scale_info['p75']:.4f}  "
              f"std={scale_info['std']:.4f}")
        print(f"  → multiplying every DA3 depth map by {scale_factor:.4f} before back-projection")

    # ── Load DA3 model (skip if fuse-only) ────────────────────────────────────
    all_pts  = []
    all_rgb  = []

    if args.fuse_only:
        print("\n--fuse-only: loading existing depth maps, skipping DA3 inference ...")
        n_with_mvc = 0
        for img_meta in images:
            stem       = Path(img_meta["name"]).stem
            depth_file = depth_dir / f"{stem}.npy"
            if not depth_file.exists():
                print(f"  [WARN] Missing: {depth_file.name} — skipping")
                continue
            cam    = cameras[img_meta["camera_id"]]
            K_orig = _intrinsics_from_camera(cam)
            w2c    = _w2c_from_image(img_meta)
            depth_orig = np.load(depth_file).astype(np.float32) * scale_factor
            orig_H, orig_W = depth_orig.shape

            # If a multi-view consistency mask exists from filter_depths_mvc.py,
            # zero out the depth where it's False so back-projection drops them.
            mvc_file = depth_file.with_suffix(".mvc.npy")
            if mvc_file.exists():
                mvc_mask = np.load(mvc_file)
                if mvc_mask.shape == depth_orig.shape:
                    depth_orig = depth_orig * mvc_mask.astype(np.float32)
                    n_with_mvc += 1

            # max_depth filter is also in COLMAP scale now
            pts = _backproject_frame(depth_orig, None, K_orig, w2c,
                                     conf_pct=args.conf_pct,
                                     max_depth=args.max_depth * scale_factor)
            if len(pts) == 0:
                continue

            img_path  = img_dir / img_meta["name"]
            img_arr   = np.array(PILImage.open(img_path).convert("RGB"), dtype=np.uint8)
            valid_mask = (depth_orig > 0.05 * scale_factor) & (depth_orig < args.max_depth * scale_factor)
            v_pix, u_pix = np.where(valid_mask)
            rgb = img_arr[v_pix, u_pix]

            per_frame_keep = max(1, (args.target * 4) // len(images))
            if len(pts) > per_frame_keep:
                rng = np.random.default_rng(hash(img_meta["name"]) & 0xFFFFFFFF)
                idx = rng.choice(len(pts), size=per_frame_keep, replace=False)
                pts, rgb = pts[idx], rgb[idx]

            all_pts.append(pts)
            all_rgb.append(rgb)
        print(f"  Loaded {len(all_pts)} frames"
              + (f" ({n_with_mvc} with MVC mask)" if n_with_mvc else ""))
    else:
        print(f"\nLoading {args.model} ...")
        from depth_anything_3.api import DepthAnything3
        model = DepthAnything3.from_pretrained(f"depth-anything/{args.model}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        print(f"  Model on {device}")

    if not args.fuse_only:
        bs = args.batch_size
        n_batches = (len(images) + bs - 1) // bs
        print(f"\nRunning DA3 on {len(images)} images in {n_batches} batch(es) of {bs} ...")

    if not args.fuse_only:
      for b_idx in range(n_batches):
        batch_imgs = images[b_idx * bs : (b_idx + 1) * bs]
        print(f"  Batch {b_idx+1}/{n_batches} ({len(batch_imgs)} images) ...", flush=True)

        # da3metric-large is a single-view model: run one image at a time.
        # We supply NO extrinsics to the model — depth is metric (metres) in
        # camera space. We back-project into world space using COLMAP poses.
        for img_meta in batch_imgs:
            img_path = img_dir / img_meta["name"]
            pil      = PILImage.open(img_path).convert("RGB")
            orig_H, orig_W = pil.height, pil.width

            cam   = cameras[img_meta["camera_id"]]
            K_orig = _intrinsics_from_camera(cam)
            w2c    = _w2c_from_image(img_meta)

            with torch.no_grad():
                pred = model.inference(
                    image       = [pil],
                    process_res = 504,
                )

            depth_proc = pred.depth[0]   # (Hd, Wd)
            conf_proc  = pred.conf[0] if pred.conf is not None else None

            # Upsample depth to original image resolution for DN-Splatter
            depth_orig = np.array(
                PILImage.fromarray(depth_proc).resize((orig_W, orig_H), PILImage.BILINEAR),
                dtype=np.float32,
            )
            conf_orig = (
                np.array(PILImage.fromarray(conf_proc).resize((orig_W, orig_H), PILImage.BILINEAR),
                         dtype=np.float32)
                if conf_proc is not None else None
            )

            # Save depth map as NPY (full resolution) — back-projection happens
            # in a second pass after we compute the global scale factor.
            stem = Path(img_meta["name"]).stem
            np.save(depth_dir / f"{stem}.npy", depth_orig)

            del pred
            torch.cuda.empty_cache()

    # ── Pass 2: compute scale + back-project all depth maps ───────────────────
    # (the fuse-only path already does this above; here we redo it for the
    # primary path now that all depth maps are on disk)
    if not args.fuse_only:
        print(f"\nDA3 inference complete — {len(images)} depth maps saved.")
        if not args.no_rescale:
            print("\nComputing global COLMAP↔DA3 scale factor ...")
            scale_factor, scale_info = compute_global_scale(
                images, cameras, points3d, _load_existing_depth,
            )
            print(f"  Pairs: {scale_info['n_pairs']:,}  from {scale_info['n_images']} images")
            print(f"  ratio (colmap/da3)  median={scale_info['median']:.4f}  "
                  f"p25={scale_info['p25']:.4f}  p75={scale_info['p75']:.4f}  "
                  f"std={scale_info['std']:.4f}")
            print(f"  → multiplying every DA3 depth map by {scale_factor:.4f} before back-projection")

        print("\nBack-projecting (scale-aligned) ...")
        for img_meta in images:
            stem       = Path(img_meta["name"]).stem
            depth_file = depth_dir / f"{stem}.npy"
            if not depth_file.exists():
                continue
            cam    = cameras[img_meta["camera_id"]]
            K_orig = _intrinsics_from_camera(cam)
            w2c    = _w2c_from_image(img_meta)
            depth_orig = np.load(depth_file).astype(np.float32) * scale_factor

            pts = _backproject_frame(
                depth_orig, None, K_orig, w2c,
                conf_pct=args.conf_pct,
                max_depth=args.max_depth * scale_factor,
            )
            if len(pts) == 0:
                continue

            img_path  = img_dir / img_meta["name"]
            img_arr   = np.array(PILImage.open(img_path).convert("RGB"), dtype=np.uint8)
            valid_mask = (depth_orig > 0.05 * scale_factor) & (depth_orig < args.max_depth * scale_factor)
            v_pix, u_pix = np.where(valid_mask)
            rgb = img_arr[v_pix, u_pix]

            per_frame_keep = max(1, (args.target * 4) // len(images))
            if len(pts) > per_frame_keep:
                rng = np.random.default_rng(hash(img_meta["name"]) & 0xFFFFFFFF)
                idx = rng.choice(len(pts), size=per_frame_keep, replace=False)
                pts, rgb = pts[idx], rgb[idx]

            all_pts.append(pts)
            all_rgb.append(rgb)

    if not args.fuse_only:
        print(f"\nBack-projection complete. Merging point clouds ...")
    else:
        print(f"\nMerging point clouds ...")

    if not all_pts:
        print("ERROR: No valid points generated.")
        sys.exit(1)

    xyz = np.concatenate(all_pts, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)
    print(f"  Total raw points: {len(xyz):,}")

    # ── Spatial outlier clipping ───────────────────────────────────────────────
    if args.sigma_clip > 0:
        mask = np.ones(len(xyz), dtype=bool)
        for ax in range(3):
            m, s = xyz[:, ax].mean(), xyz[:, ax].std()
            mask &= np.abs(xyz[:, ax] - m) <= args.sigma_clip * s
        before = len(xyz)
        xyz, rgb = xyz[mask], rgb[mask]
        print(f"  Outlier clip ({args.sigma_clip}σ): removed {before-len(xyz):,}, "
              f"{len(xyz):,} remain")

    # ── Scene AABB ─────────────────────────────────────────────────────────────
    bounds = {
        "x_min": float(xyz[:, 0].min()), "x_max": float(xyz[:, 0].max()),
        "y_min": float(xyz[:, 1].min()), "y_max": float(xyz[:, 1].max()),
        "z_min": float(xyz[:, 2].min()), "z_max": float(xyz[:, 2].max()),
    }
    vol = ((bounds["x_max"] - bounds["x_min"]) *
           (bounds["y_max"] - bounds["y_min"]) *
           (bounds["z_max"] - bounds["z_min"]))
    print(f"  Scene AABB  : "
          f"x=[{bounds['x_min']:.2f},{bounds['x_max']:.2f}]  "
          f"y=[{bounds['y_min']:.2f},{bounds['y_max']:.2f}]  "
          f"z=[{bounds['z_min']:.2f},{bounds['z_max']:.2f}]")
    print(f"  Volume      : {vol:.1f} cubic units")

    # Embed the scale factor + diagnostics so downstream code can verify alignment
    bounds["scale_factor_da3_to_colmap"] = float(scale_factor)
    bounds["scale_aligned"] = bool(not args.no_rescale)
    if scale_info is not None:
        bounds["scale_info"] = scale_info

    with open(out_bounds, "w") as f:
        json.dump(bounds, f, indent=2)
    print(f"  Bounds saved: {out_bounds}")

    # ── Voxel subsample ────────────────────────────────────────────────────────
    # Estimate voxel size to hit the target point count
    if len(xyz) > args.target:
        # Iterative bisection to find voxel size that gives ~target points
        lo, hi = 0.0, float(max(
            bounds["x_max"] - bounds["x_min"],
            bounds["y_max"] - bounds["y_min"],
            bounds["z_max"] - bounds["z_min"],
        ))
        for _ in range(20):
            mid_v = (lo + hi) / 2
            _, idx_v = np.unique(np.floor(xyz / mid_v).astype(np.int32), axis=0, return_index=True)
            if len(idx_v) > args.target:
                lo = mid_v
            else:
                hi = mid_v
        # Apply final voxel size
        _, idx_v = np.unique(np.floor(xyz / hi).astype(np.int32), axis=0, return_index=True)
        xyz, rgb = xyz[idx_v], rgb[idx_v]
        print(f"  Voxel subsampled to {len(xyz):,} points (voxel={hi:.4f})")
    else:
        print(f"  Already ≤ {args.target:,} points — no subsampling needed")

    # ── Write PLY ──────────────────────────────────────────────────────────────
    ply_bytes = _build_ply(xyz, rgb)
    with open(out_ply, "wb") as f:
        f.write(ply_bytes)
    size_mb = out_ply.stat().st_size / 1_048_576
    print(f"\n  da3_init.ply : {len(xyz):,} points  ({size_mb:.0f} MB) → {out_ply}")

    # ── Summary ────────────────────────────────────────────────────────────────
    depth_files = list(depth_dir.glob("*.npy"))
    print(f"  Depth maps   : {len(depth_files)} files → {depth_dir}")
    print("\nDA3 depth generation complete.")


if __name__ == "__main__":
    main()
