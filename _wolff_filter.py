"""Wolff et al. (2016) — Point Cloud Noise and Outlier Removal for
Image-Based 3D Reconstruction. Implemented from the paper (Section 3),
since the original authors did not release code and the algorithm is
covered by Disney patent US10074160B2 (paper has full math, so this is
a clean-room implementation for academic comparison).

Algorithm summary (paper Section 3):

For each depth map D_i, back-project pixels to a 3D point set. Each
view's depth map is itself a "range surface" -- a triangulated mesh of
depth values. For each candidate point p originating from any view:

  1. Project p into every other view j.
  2. Look up the range-surface depth z_i(p) at that pixel.
  3. Signed distance d_j(p) = z_j(p) - z_p_camera (eq. 1).
     - d_j < 0 means p is behind view j's surface -> view j can't see p
       (skip via indicator I^G_sigma).
     - d_j > sigma is truncated to sigma in the average (eq. 4).
  4. View-angle weight w_j = max(0, n(p) . (p - v_j) / |p - v_j|) (eq. 3).
     Only include views with v_j^T v_i > 0 (drop opposite sides).
  5. Weighted average signed distance d(p) (eq. 4).
  6. Photometric: count views v(p) where -sigma < d_j < sigma (eq. 6),
     and the std-dev p(p) of colors at those intersection points (eq. 7).
  7. Keep p iff -t_d < d(p) < 0 AND p(p) < t_p AND v(p) > t_v (eq. 8).

Parameters (paper Section 4):
  sigma = 1% of scene depth range
  t_d   = 0.1 * sigma
  t_v   = 7.5% of view count
  t_p   = 0.2

Inputs we use:
  - Femto Mega ToF .npy depth maps (1920x1080, metric metres, float32)
  - aligned RGB images
  - COLMAP poses (OpenGL c2w, COLMAP-unit translations)
  - tof_bounds.json for scale conversion to metric

Output:
  - Filtered point cloud .ply (positions + colors + normals)
  - Optional: Poisson surface mesh via Open3D (.ply textured by vertex colors)
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


_GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def load_views(project_root: Path, max_views: int = 0, view_stride: int = 1):
    """Load (depth, image, K, w2c, c2w, name) per view. Returns list of dicts."""
    tj = project_root / "colmap/dense/transforms.json"
    bj = project_root / "colmap/dense/tof_bounds.json"
    meta = json.loads(tj.read_text())
    bounds = json.loads(bj.read_text())
    s_factor = float(bounds["scale_factor_da3_to_colmap"])
    colmap_to_metric = 1.0 / s_factor

    fx = float(meta.get("fl_x", 1125.36))
    fy = float(meta.get("fl_y", 1125.07))
    cx = float(meta.get("cx", 966.03))
    cy = float(meta.get("cy", 519.94))
    W = int(meta.get("w", 1920)); H = int(meta.get("h", 1080))
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    frames = meta["frames"]
    if view_stride > 1:
        frames = frames[::view_stride]
    if max_views > 0:
        frames = frames[:max_views]

    views = []
    for frame in frames:
        fp = Path(frame["file_path"])
        name = fp.stem                                          # IMG_femto_XXXX
        img_path = (project_root / "colmap/dense" / fp).resolve()
        if not img_path.exists():
            img_path = (project_root / "nerfstudio_data" / fp).resolve()
        depth_path = project_root / "nerfstudio_data/depths_femto" / f"{name}.npy"
        if not (img_path.exists() and depth_path.exists()):
            continue
        c2w_gl = np.array(frame["transform_matrix"], dtype=np.float64)
        c2w = c2w_gl @ _GL_TO_CV
        c2w[:3, 3] *= colmap_to_metric
        w2c = np.linalg.inv(c2w)
        views.append({
            "name": name,
            "depth_path": depth_path,
            "img_path": img_path,
            "K": K, "W": W, "H": H,
            "w2c": w2c, "c2w": c2w,
            "cam_pos": c2w[:3, 3],
            "view_dir": c2w[:3, 2] / (np.linalg.norm(c2w[:3, 2]) + 1e-12),
        })
    print(f"[wolff] loaded {len(views)} views (scale_factor={s_factor:.3f})")
    return views


def back_project_view(view, pixel_stride: int = 1):
    """Back-project a depth map's pixels into 3D world space.

    Returns:
        pts  : (N, 3) world-space metric points
        rgb  : (N, 3) uint8 colors
        norm : (N, 3) per-point normals (unit, from depth-map gradients)
        pix  : (N, 2) original (px, py) so we can index the source depth
    """
    depth = np.load(view["depth_path"]).astype(np.float32)
    img = np.array(Image.open(view["img_path"]).convert("RGB"))
    if img.shape[:2] != depth.shape:
        # Resize image to depth resolution (color and depth share the same
        # intrinsics per femto_intrinsics.json so this rarely fires).
        img = np.array(Image.fromarray(img).resize(
            (depth.shape[1], depth.shape[0]), Image.BILINEAR
        ))
    H, W = depth.shape
    K = view["K"]
    c2w = view["c2w"]

    # Pixel grid (downsampled)
    yy, xx = np.mgrid[0:H:pixel_stride, 0:W:pixel_stride]
    valid = depth[yy, xx] > 0.05
    if not valid.any():
        return (np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8),
                np.zeros((0, 3)), np.zeros((0, 2), dtype=np.int32))
    yy = yy[valid]; xx = xx[valid]
    z = depth[yy, xx]

    # Camera-space rays
    x_cam = (xx - K[0, 2]) / K[0, 0] * z
    y_cam = (yy - K[1, 2]) / K[1, 1] * z
    pts_cam = np.stack([x_cam, y_cam, z], axis=-1)              # (N, 3) OpenCV
    # To world
    homog = np.hstack([pts_cam, np.ones((len(pts_cam), 1))])
    pts_world = (c2w @ homog.T).T[:, :3]

    rgb = img[yy, xx]                                           # (N, 3)

    # Per-pixel normals via depth-map gradient (image-space neighbors, paper eq. 3)
    # Use Sobel-like 2-pixel finite differences in image space.
    # We approximate the world-normal as the cross product of the camera-space
    # gradients in x and y. Robust enough per paper, "we found this method to
    # be fast and sufficient".
    norm = np.zeros_like(pts_world)
    if len(pts_world) > 4:
        # Build full-res 3D camera-space grid (sparse, but enough for grads)
        cam_full = np.zeros((H, W, 3), dtype=np.float32)
        d_safe = np.where(depth > 0.05, depth, np.nan)
        y_grid, x_grid = np.mgrid[0:H, 0:W]
        cam_full[..., 0] = (x_grid - K[0, 2]) / K[0, 0] * d_safe
        cam_full[..., 1] = (y_grid - K[1, 2]) / K[1, 1] * d_safe
        cam_full[..., 2] = d_safe
        # Sobel-style central diff
        dy = np.roll(cam_full, -1, axis=0) - np.roll(cam_full, 1, axis=0)
        dx = np.roll(cam_full, -1, axis=1) - np.roll(cam_full, 1, axis=1)
        n_cam = np.cross(dx, dy)                                # (H, W, 3)
        n_cam /= (np.linalg.norm(n_cam, axis=-1, keepdims=True) + 1e-12)
        # Orient towards camera (negative z in camera space)
        flip = n_cam[..., 2] > 0
        n_cam[flip] *= -1
        # To world
        R = c2w[:3, :3]
        n_world_full = n_cam @ R.T
        norm = n_world_full[yy, xx].astype(np.float64)
        bad = ~np.isfinite(norm).all(axis=-1)
        norm[bad] = 0
        norm = norm / (np.linalg.norm(norm, axis=-1, keepdims=True) + 1e-12)

    pix = np.stack([xx, yy], axis=-1).astype(np.int32)
    return pts_world, rgb, norm, pix


def wolff_filter(views, sigma_frac: float = 0.01,
                 td_frac: float = 0.1, tv_frac: float = 0.075,
                 tp_thresh: float = 0.2,
                 pixel_stride: int = 4,
                 verbose: bool = True):
    """Run Wolff filter across all views. Returns filtered point cloud arrays.

    pixel_stride : sample every Nth pixel when back-projecting. Cuts point
                   count by stride^2 (stride=4 ~16x speedup, ~80M -> 5M pts).
                   Cross-view lookups still query the *full* depth maps, so
                   accuracy of the consistency test is preserved.
    """
    # ----- Back-project every view -----
    all_pts, all_rgb, all_norm, view_owner = [], [], [], []
    print(f"\n[wolff] back-projecting {len(views)} depth maps (pixel_stride={pixel_stride}) ...", flush=True)
    t0 = time.time()
    for i, v in enumerate(views):
        pts, rgb, norm, _ = back_project_view(v, pixel_stride=pixel_stride)
        all_pts.append(pts); all_rgb.append(rgb); all_norm.append(norm)
        view_owner.append(np.full(len(pts), i, dtype=np.int32))
        if verbose and (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(views)}  +{len(pts):,} pts  total={sum(map(len,all_pts)):,}", flush=True)
    pts = np.concatenate(all_pts)
    rgb = np.concatenate(all_rgb)
    norm = np.concatenate(all_norm)
    owner = np.concatenate(view_owner)
    print(f"[wolff] {len(pts):,} candidate points "
          f"(back-projection {time.time()-t0:.1f}s)")

    # ----- Pre-load depth maps + RGB to RAM for fast lookup -----
    depth_maps, color_maps = [], []
    print(f"[wolff] caching depth + color per view ...")
    for v in views:
        depth_maps.append(np.load(v["depth_path"]).astype(np.float32))
        img = np.array(Image.open(v["img_path"]).convert("RGB"))
        if img.shape[:2] != depth_maps[-1].shape:
            img = np.array(Image.fromarray(img).resize(
                (depth_maps[-1].shape[1], depth_maps[-1].shape[0]), Image.BILINEAR
            ))
        color_maps.append(img.astype(np.float32) / 255.0)

    # ----- Determine sigma from scene depth range -----
    z_range = float(np.percentile(pts[:, 2], 99) - np.percentile(pts[:, 2], 1))
    z_range = max(z_range, 0.5)
    sigma = sigma_frac * z_range
    td = td_frac * sigma
    tv = max(2, int(tv_frac * len(views)))
    print(f"[wolff] sigma={sigma:.4f}m  td={td:.4f}m  tv={tv}  tp={tp_thresh}")

    # ----- Per-point consistency: vectorise across points, loop views -----
    # Allocate accumulators
    N = len(pts)
    w_sum = np.zeros(N, dtype=np.float64)           # numerator (weighted distance)
    w_total = np.zeros(N, dtype=np.float64)         # denominator (sum of weights)
    visibility = np.zeros(N, dtype=np.int32)        # number of views passing photometric band
    color_sum = np.zeros((N, 3), dtype=np.float64)  # for std-dev
    color_sq = np.zeros((N, 3), dtype=np.float64)

    print(f"\n[wolff] cross-view consistency pass ...")
    t0 = time.time()
    sigma_f32 = np.float32(sigma)
    for j, vj in enumerate(views):
        # Skip projecting points into the view that *owns* them later --
        # for simplicity we still project but at d=0 they auto-pass
        # (paper Sec 3.1 explicitly says include self).
        # Cull: only views looking the same-ish direction as the avg
        # (we do per-point culling via head-on weight; cheap to skip here)
        Kj = vj["K"]; w2c_j = vj["w2c"]; v_pos = vj["cam_pos"]
        Hj, Wj = depth_maps[j].shape

        homog = np.hstack([pts, np.ones((N, 1))])
        p_cam = (w2c_j @ homog.T).T[:, :3]
        depth_j_at_p = p_cam[:, 2]                       # depth of candidate in view j
        in_front = depth_j_at_p > 0.05
        d_safe = np.where(in_front, depth_j_at_p, 0.1)
        px = (Kj[0, 0] * p_cam[:, 0] / d_safe + Kj[0, 2])
        py = (Kj[1, 1] * p_cam[:, 1] / d_safe + Kj[1, 2])
        ix = np.clip(np.round(px).astype(np.int32), 0, Wj - 1)
        iy = np.clip(np.round(py).astype(np.int32), 0, Hj - 1)
        in_bounds = (px >= 0) & (px < Wj) & (py >= 0) & (py < Hj)
        valid = in_front & in_bounds

        # Look up the depth-map's depth at that pixel (the range-surface
        # depth z_j(p) per paper eq. 1)
        z_at_pix = depth_maps[j][iy, ix].astype(np.float32)
        has_depth = (z_at_pix > 0.05) & valid

        # Signed distance d_j = z_j(p) - z_p_in_j      (paper eq. 1)
        # (z_j(p) is the range-surface depth at the projected pixel,
        # depth_j_at_p is the depth of the candidate -- so the sign
        # convention: negative = p is BEHIND the surface, positive = in front).
        # NB: paper writes d_i(p) = z_i(p) - z. We match.
        d_j = z_at_pix - depth_j_at_p.astype(np.float32)

        # View-angle weight w_j = n(p) . unit(p - v_j)  (paper eq. 3)
        # n(p) is candidate's normal; v_j is camera position.
        ray = pts - v_pos[None, :]
        ray_n = ray / (np.linalg.norm(ray, axis=-1, keepdims=True) + 1e-12)
        w_j = np.einsum("ij,ij->i", norm, ray_n)
        w_j = np.clip(w_j, 0.0, None)

        # Cull views that look at p from opposite side (vj_dir . vi_dir < 0)
        vi_dir = np.array([views[i]["view_dir"] for i in owner])
        vj_dir = vj["view_dir"]
        same_side = (vi_dir @ vj_dir) > 0
        w_j = np.where(same_side, w_j, 0.0)

        # Geometric indicator (eq. 2): contribute if not too far BEHIND surface
        # We want -sigma < d_j (so d_j > -sigma). Truncate positive at sigma.
        I_G = (d_j > -sigma_f32) & has_depth & (w_j > 0)
        d_trunc = np.where(d_j > sigma_f32, sigma_f32, d_j)
        contrib_num = (I_G.astype(np.float32) * w_j.astype(np.float32) * d_trunc)
        contrib_den = (I_G.astype(np.float32) * w_j.astype(np.float32))
        w_sum += contrib_num.astype(np.float64)
        w_total += contrib_den.astype(np.float64)

        # Photometric band: only count views where |d_j| < sigma (eq. 5)
        I_P = (np.abs(d_j) < sigma_f32) & has_depth
        visibility += I_P.astype(np.int32)
        # Sample interpolated color (just nearest for speed)
        c = color_maps[j][iy, ix]  # (N, 3)
        c = np.where(I_P[:, None], c, 0.0)
        color_sum += c.astype(np.float64)
        color_sq += (c.astype(np.float64) ** 2)

        if verbose and (j + 1) % 5 == 0:
            print(f"  {j+1}/{len(views)}  ({time.time()-t0:.1f}s elapsed)", flush=True)

    print(f"[wolff] cross-view pass done in {time.time()-t0:.1f}s")

    # ----- Aggregate -----
    eps = 1e-9
    d_avg = w_sum / np.maximum(w_total, eps)
    color_mean = color_sum / np.maximum(visibility[:, None], 1)
    color_var = (color_sq / np.maximum(visibility[:, None], 1)) - color_mean ** 2
    color_var = np.maximum(color_var, 0)
    color_std = np.sqrt(color_var.sum(axis=-1))                  # eq. 7

    # ----- Threshold (eq. 8) -----
    keep = (d_avg > -td) & (d_avg < 0) & (color_std < tp_thresh) & (visibility > tv)
    print(f"[wolff] kept {int(keep.sum()):,} of {N:,} "
          f"({100*keep.mean():.1f}%)")
    return pts[keep], rgb[keep], norm[keep]


def write_ply(path: Path, pts, rgb, norm=None):
    n = len(pts)
    has_n = norm is not None and len(norm) == n
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_n:
            f.write("property float nx\nproperty float ny\nproperty float nz\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            row = f"{pts[i,0]:.5f} {pts[i,1]:.5f} {pts[i,2]:.5f}"
            if has_n:
                row += f" {norm[i,0]:.4f} {norm[i,1]:.4f} {norm[i,2]:.4f}"
            row += f" {int(rgb[i,0])} {int(rgb[i,1])} {int(rgb[i,2])}\n"
            f.write(row)
    print(f"[wolff] wrote {path} ({path.stat().st_size/1_048_576:.1f} MB)")


def poisson_mesh(pts, norm, rgb, depth=9):
    """Open3D screened Poisson surface reconstruction."""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.normals = o3d.utility.Vector3dVector(norm)
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255.0)
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=1.1, linear_fit=False,
    )
    return mesh


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=Path("."))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--view-stride", type=int, default=3,
                   help="Use every Nth view (3 gives ~46 of 140).")
    p.add_argument("--max-views", type=int, default=0)
    p.add_argument("--pixel-stride", type=int, default=4,
                   help="Back-projection pixel stride (default 4). Lower = "
                        "denser points + slower. 4 -> ~5M candidate points "
                        "across 46 views.")
    p.add_argument("--sigma-frac", type=float, default=0.01,
                   help="sigma as fraction of scene depth range (paper default 1%)")
    p.add_argument("--td-frac", type=float, default=0.1)
    p.add_argument("--tv-frac", type=float, default=0.075)
    p.add_argument("--tp", type=float, default=0.2)
    p.add_argument("--poisson-depth", type=int, default=9)
    p.add_argument("--no-mesh", action="store_true",
                   help="Skip Poisson meshing (just dump filtered cloud)")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    views = load_views(args.project_root, max_views=args.max_views,
                       view_stride=args.view_stride)
    if len(views) < 5:
        raise SystemExit("Not enough views found")

    pts, rgb, norm = wolff_filter(
        views,
        sigma_frac=args.sigma_frac, td_frac=args.td_frac,
        tv_frac=args.tv_frac, tp_thresh=args.tp,
        pixel_stride=args.pixel_stride,
    )

    ply_path = args.output_dir / "wolff_filtered_points.ply"
    write_ply(ply_path, pts, rgb, norm)

    if not args.no_mesh:
        print(f"\n[wolff] Poisson surface reconstruction (depth={args.poisson_depth}) ...")
        t0 = time.time()
        mesh = poisson_mesh(pts, norm, rgb, depth=args.poisson_depth)
        import open3d as o3d
        mesh_path = args.output_dir / "wolff_poisson_mesh.ply"
        o3d.io.write_triangle_mesh(str(mesh_path), mesh)
        print(f"[wolff] Poisson done in {time.time()-t0:.1f}s -> "
              f"{len(mesh.triangles):,} faces ({mesh_path})")


if __name__ == "__main__":
    main()
