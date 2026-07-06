"""Track 2a (modified, post-Track-1) of the COMPREHENSIVE_PLAN — seed a
NN-copied desk patch as Gaussians on the desk plane under each object.

Track 1's measurement showed all V32 objects have <= 11% truly-unseen
footprint. NN-copy from observed views is sufficient — no diffusion
inpainter (LaMa / GSFix3D) needed for this dataset.

This script combines what the COMPREHENSIVE_PLAN called M1+M2+M3:

  M1 (mesh raycast):  uses the Track-1 NPZ (grid points already on the
                      desk plane, with per-point view counts).
  M2 (RGB inpaint):   per-point colour = sample from the best-quality
                      view that observed it; truly-unseen points NN-copy
                      colour from the nearest well-observed grid point.
                      No LaMa needed at V32's <30% unseen regime.
  M3 (unproject):     transform metric grid -> nerfstudio splat space
                      via the trained dataparser_transforms.json; seed
                      one flat-on-plane Gaussian per grid point with
                      computed SH-DC colour, high opacity, and rotation
                      that aligns the local Gaussian normal to the desk
                      normal in splat space.

  M4 (local opt):     deferred — first test whether seeded Gaussians
                      render correctly without retraining.

Output per object: `<out>/desk_patch_<slug>.ply` (nerfstudio splat
PLY schema with SH degree 3 zero-padded; drop-in additive to the V32
scene splat).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from plyfile import PlyData, PlyElement

sys.path.insert(0, str(Path(__file__).parent))
from scene_segmenter.views import V32ViewSource  # noqa: E402

SH_C0 = 0.28209479177387814

# nerfstudio Gaussian splat PLY property list. Must match the existing
# V32 splat schema exactly so we can concatenate.
PLY_FIELDS = (
    ["x", "y", "z", "nx", "ny", "nz",
     "f_dc_0", "f_dc_1", "f_dc_2"]
    + [f"f_rest_{i}" for i in range(45)]      # SH degree 3 (15 coefs × 3)
    + ["opacity", "scale_0", "scale_1", "scale_2",
       "rot_0", "rot_1", "rot_2", "rot_3"]
)


def _load_dataparser(dp_path: Path):
    d = json.loads(dp_path.read_text())
    T = np.asarray(d["transform"], dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    s = float(d["scale"])
    return R, t, s


def _load_colmap_to_metric(bounds_path: Path) -> float:
    b = json.loads(bounds_path.read_text())
    return 1.0 / float(b["scale_factor_da3_to_colmap"])


def _quat_y_to(v_dir: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) rotating local +Y axis to unit `v_dir`.

    nerfstudio quat convention: (w, x, y, z), unit-norm.
    Returns identity (1, 0, 0, 0) if `v_dir` already ~ (0, 1, 0).
    """
    v = v_dir / max(np.linalg.norm(v_dir), 1e-12)
    y = np.array([0.0, 1.0, 0.0])
    c = float(np.dot(y, v))
    if c > 0.999999:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if c < -0.999999:
        # 180° rotation about X axis
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    axis = np.cross(y, v)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    angle = float(np.arccos(c))
    half = angle * 0.5
    sin_h = float(np.sin(half))
    w = float(np.cos(half))
    return np.array([w, axis[0] * sin_h, axis[1] * sin_h, axis[2] * sin_h],
                    dtype=np.float32)


def _metric_to_splat(positions_metric: np.ndarray,
                     R: np.ndarray, t: np.ndarray,
                     dp_scale: float, colmap_to_metric: float) -> np.ndarray:
    """Forward transform metric metres -> splat space.

    metric -> COLMAP units (/ colmap_to_metric, since colmap_to_metric =
    1/scale_factor_da3_to_colmap), then COLMAP -> splat via dp_scale * (R @ p + t).
    """
    colmap_pos = positions_metric / colmap_to_metric
    splat_pos = dp_scale * (colmap_pos @ R.T + t[None, :])
    return splat_pos.astype(np.float32)


def _best_view_per_point(view_source, scene_mesh_path: Path,
                         points_metric: np.ndarray,
                         seen_count: np.ndarray,
                         depth_tol: float = 0.015,
                         sam3_masks: dict | None = None,
                         desk_normal_metric: np.ndarray | None = None):
    """For each grid point, find the best view that ACTUALLY observed it
    (unobstructed line of sight) and return (view_idx, u, v) per point.

    Score = (sum of cos(angle off normal) over seen views) -- prefers
    views that look more down at the desk than grazing-along-it.

    SAM3 OBJECT-MASK GUARD (`sam3_masks`):
      Even if the scene mesh raycast says no obstruction, OpenMVS sometimes
      has holes in the OBJECT mesh (e.g. lobster claws not fully
      reconstructed). Rays pass through those mesh holes -> visibility check
      falsely passes -> color sampled from the OBJECT's pixel in the photo
      instead of the desk's. Result: lobster-red patches.

      Fix: if `sam3_masks[view_stem][v, u] > 128` (the object is at that pixel
      in the actual photo), drop that view as a color source for that point.
    """
    import open3d as o3d
    import trimesh as _tm
    mesh = _tm.load(str(scene_mesh_path), force="mesh", process=False)
    tm = o3d.t.geometry.TriangleMesh()
    tm.vertex.positions = o3d.core.Tensor(
        np.asarray(mesh.vertices, dtype=np.float32))
    tm.triangle.indices = o3d.core.Tensor(
        np.asarray(mesh.faces, dtype=np.uint32))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tm)

    views = view_source.all_views()
    n_views = len(views)
    n_pts = len(points_metric)
    if desk_normal_metric is None:
        desk_normal_metric = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    desk_normal_metric = (desk_normal_metric
                          / max(np.linalg.norm(desk_normal_metric), 1e-9))
    best_score = np.full(n_pts, -1.0, dtype=np.float32)
    best_view = np.full(n_pts, -1, dtype=np.int32)
    best_uv = np.full((n_pts, 2), -1.0, dtype=np.float32)
    sam3_drops_total = 0

    for vi, view in enumerate(views):
        cam_pos = view.c2w[:3, 3].astype(np.float32)
        dirs = points_metric - cam_pos[None, :]
        dists = np.linalg.norm(dirs, axis=-1)
        dirs_norm = dirs / np.maximum(dists[:, None], 1e-9)
        K = view.K
        w2c = view.w2c
        pts_cam = (w2c[:3, :3] @ points_metric.T).T + w2c[:3, 3]
        zc = pts_cam[:, 2]
        in_front = zc > 0.05
        u = K[0, 0] * pts_cam[:, 0] / np.where(in_front, zc, 1.0) + K[0, 2]
        v = K[1, 1] * pts_cam[:, 1] / np.where(in_front, zc, 1.0) + K[1, 2]
        in_frame = (u >= 0) & (u < view.width) & (v >= 0) & (v < view.height)
        # Visibility test (ray unobstructed)
        origins = np.broadcast_to(cam_pos, dirs_norm.shape)
        rays_np = np.concatenate([origins, dirs_norm], axis=-1).astype(np.float32)
        ans = scene.cast_rays(o3d.core.Tensor(rays_np))
        hit_t = ans["t_hit"].numpy()
        visible = in_front & in_frame & (hit_t >= dists - depth_tol)

        # SAM3 OBJECT-MASK GUARD ------------------------------------------------
        if sam3_masks is not None:
            mask = sam3_masks.get(view.name)
            if mask is not None:
                mh, mw = mask.shape[:2]
                # Sample the mask only where we'd otherwise sample colour.
                # Need integer (u, v) and clipped to mask bounds.
                u_int = np.clip(np.round(u).astype(np.int32), 0, mw - 1)
                v_int = np.clip(np.round(v).astype(np.int32), 0, mh - 1)
                # Only meaningful where in_frame; for out-of-frame use 0 to avoid index issues.
                u_int = np.where(in_frame, u_int, 0)
                v_int = np.where(in_frame, v_int, 0)
                pix = mask[v_int, u_int]
                inside_object = (pix > 128) & in_frame
                pre_visible = visible.sum()
                visible = visible & ~inside_object
                sam3_drops_total += int(pre_visible - visible.sum())
        # -----------------------------------------------------------------------

        # Score = cos(angle between viewing dir and desk normal); want top-down.
        # Use the per-object FITTED normal if available -- a tilted desk's
        # "top-down" view is also tilted in metric world.
        cos_off = np.abs(dirs_norm @ desk_normal_metric)
        score = (cos_off.astype(np.float32)
                 * (1.0 / np.maximum(dists, 0.1)).astype(np.float32))
        better = visible & (score > best_score)
        if better.any():
            best_score[better] = score[better]
            best_view[better] = vi
            best_uv[better, 0] = u[better]
            best_uv[better, 1] = v[better]

    if sam3_masks is not None:
        print(f"    SAM3 guard rejected {sam3_drops_total:,} "
              f"(point, view) pairs where photo shows the object")
    return best_view, best_uv


def _topk_views_per_point(view_source, scene_mesh_path: Path,
                          points_metric: np.ndarray,
                          seen_count: np.ndarray,
                          depth_tol: float = 0.015,
                          sam3_masks: dict | None = None,
                          desk_normal_metric: np.ndarray | None = None,
                          k: int = 5):
    """Like `_best_view_per_point` but keeps the TOP-K visible views per point.

    Top-K + median-color is robust against:
      - residual edge bleed from SAM3 mask anti-aliasing (one bad view → outlier)
      - lighting variance / specular highlights across the orbit
    Returns:
      topk_view: (n_pts, k) int32, -1 for empty slots
      topk_uv:   (n_pts, k, 2) float32, NaN for empty slots
    """
    import open3d as o3d
    import trimesh as _tm
    mesh = _tm.load(str(scene_mesh_path), force="mesh", process=False)
    tm = o3d.t.geometry.TriangleMesh()
    tm.vertex.positions = o3d.core.Tensor(
        np.asarray(mesh.vertices, dtype=np.float32))
    tm.triangle.indices = o3d.core.Tensor(
        np.asarray(mesh.faces, dtype=np.uint32))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tm)

    views = view_source.all_views()
    n_views = len(views)
    n_pts = len(points_metric)
    if desk_normal_metric is None:
        desk_normal_metric = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    desk_normal_metric = (desk_normal_metric
                          / max(np.linalg.norm(desk_normal_metric), 1e-9))

    # Maintain top-K: store as (n_pts, k) arrays sorted ascending by score.
    # New score replaces position 0 (lowest) and we re-sort the row.
    topk_score = np.full((n_pts, k), -1.0, dtype=np.float32)
    topk_view = np.full((n_pts, k), -1, dtype=np.int32)
    topk_uv = np.full((n_pts, k, 2), np.nan, dtype=np.float32)

    sam3_drops_total = 0
    for vi, view in enumerate(views):
        cam_pos = view.c2w[:3, 3].astype(np.float32)
        dirs = points_metric - cam_pos[None, :]
        dists = np.linalg.norm(dirs, axis=-1)
        dirs_norm = dirs / np.maximum(dists[:, None], 1e-9)
        K = view.K
        w2c = view.w2c
        pts_cam = (w2c[:3, :3] @ points_metric.T).T + w2c[:3, 3]
        zc = pts_cam[:, 2]
        in_front = zc > 0.05
        u = K[0, 0] * pts_cam[:, 0] / np.where(in_front, zc, 1.0) + K[0, 2]
        v = K[1, 1] * pts_cam[:, 1] / np.where(in_front, zc, 1.0) + K[1, 2]
        in_frame = (u >= 0) & (u < view.width) & (v >= 0) & (v < view.height)
        origins = np.broadcast_to(cam_pos, dirs_norm.shape)
        rays_np = np.concatenate([origins, dirs_norm], axis=-1).astype(np.float32)
        ans = scene.cast_rays(o3d.core.Tensor(rays_np))
        hit_t = ans["t_hit"].numpy()
        visible = in_front & in_frame & (hit_t >= dists - depth_tol)

        if sam3_masks is not None:
            mask = sam3_masks.get(view.name)
            if mask is not None:
                mh, mw = mask.shape[:2]
                u_int = np.clip(np.round(u).astype(np.int32), 0, mw - 1)
                v_int = np.clip(np.round(v).astype(np.int32), 0, mh - 1)
                u_int = np.where(in_frame, u_int, 0)
                v_int = np.where(in_frame, v_int, 0)
                pix = mask[v_int, u_int]
                inside_object = (pix > 128) & in_frame
                pre = visible.sum()
                visible = visible & ~inside_object
                sam3_drops_total += int(pre - visible.sum())

        cos_off = np.abs(dirs_norm @ desk_normal_metric)
        score_v = (cos_off.astype(np.float32)
                   * (1.0 / np.maximum(dists, 0.1)).astype(np.float32))
        score_v = np.where(visible, score_v, -1.0)

        # For each point where new score > topk_score[:, 0] (the lowest of the K),
        # replace position 0 with the new view and re-sort the row.
        better = score_v > topk_score[:, 0]
        idx = np.where(better)[0]
        if len(idx) > 0:
            topk_score[idx, 0] = score_v[idx]
            topk_view[idx, 0] = vi
            topk_uv[idx, 0, 0] = u[idx].astype(np.float32)
            topk_uv[idx, 0, 1] = v[idx].astype(np.float32)
            # Re-sort each updated row by score ASC so position 0 is the lowest again.
            order = np.argsort(topk_score[idx], axis=1)
            row_idx = np.arange(len(idx))[:, None]
            topk_score[idx] = topk_score[idx][row_idx, order]
            topk_view[idx] = topk_view[idx][row_idx, order]
            topk_uv[idx] = topk_uv[idx][row_idx, order]

    if sam3_masks is not None:
        print(f"    SAM3 guard rejected {sam3_drops_total:,} (point, view) pairs")
    n_with_any = int((topk_view[:, -1] >= 0).sum())  # row's max-score slot filled
    n_with_full_k = int((topk_view[:, 0] >= 0).sum())
    print(f"    found >=1 visible view for {n_with_any:,}/{n_pts:,} points, "
          f"full {k}-view stack for {n_with_full_k:,}")
    return topk_view, topk_uv


def _sample_color_topk(view_source, topk_view: np.ndarray, topk_uv: np.ndarray
                       ) -> np.ndarray:
    """Sample RGB per-point as the per-channel MEDIAN of the top-K views.

    For each point row, looks at all filled slots (view >= 0), samples RGB
    via bilinear interp from each view's photo at its (u, v), then medians
    those K samples per channel. Median is robust to a single bleed-through.

    Returns (n_pts, 3) in [0, 1]; NaN for points with no visible view.
    """
    import cv2
    views = view_source.all_views()
    n_pts, k = topk_view.shape
    # Preload only the views we touched
    used = set(int(vi) for vi in np.unique(topk_view) if vi >= 0)
    photo_cache: dict[int, np.ndarray] = {}
    for vi in used:
        img = cv2.imread(str(views[vi].image_path), cv2.IMREAD_COLOR)
        if img is not None:
            photo_cache[vi] = img[:, :, ::-1]  # BGR -> RGB

    out = np.full((n_pts, 3), np.nan, dtype=np.float32)
    for i in range(n_pts):
        per_view_rgb = []
        for j in range(k):
            vi = int(topk_view[i, j])
            if vi < 0:
                continue
            img = photo_cache.get(vi)
            if img is None:
                continue
            u, v = float(topk_uv[i, j, 0]), float(topk_uv[i, j, 1])
            H, W = img.shape[:2]
            x0 = int(np.floor(u))
            y0 = int(np.floor(v))
            x1, y1 = x0 + 1, y0 + 1
            x0c = max(0, min(W - 1, x0))
            y0c = max(0, min(H - 1, y0))
            x1c = max(0, min(W - 1, x1))
            y1c = max(0, min(H - 1, y1))
            fx, fy = u - x0, v - y0
            c00 = img[y0c, x0c].astype(np.float32)
            c01 = img[y0c, x1c].astype(np.float32)
            c10 = img[y1c, x0c].astype(np.float32)
            c11 = img[y1c, x1c].astype(np.float32)
            c = (c00 * (1 - fx) * (1 - fy) + c01 * fx * (1 - fy)
                 + c10 * (1 - fx) * fy + c11 * fx * fy)
            per_view_rgb.append(c / 255.0)
        if per_view_rgb:
            out[i] = np.median(np.stack(per_view_rgb, axis=0), axis=0)
    return out


def _load_sam3_masks(sam3_cache: Path, prompt_slug: str,
                     view_names: list[str], dilate_px: int = 8) -> dict:
    """Load per-view SAM3 binary masks for one prompt as
    {view_stem: HxW uint8 mask}. Returns empty dict if cache missing.

    `dilate_px`: morphological dilation in pixels. Catches the anti-aliased
    soft-edge halo around each SAM3 object — those edge pixels still contain
    object colour even though the binary mask says "outside". 8 px on a
    1920x1080 / 1 m scene is ~4 mm of safety margin.
    """
    import cv2
    masks = {}
    if not sam3_cache.is_dir():
        print(f"    WARN SAM3 cache {sam3_cache} missing -- skipping mask guard")
        return masks
    k = None
    if dilate_px > 0:
        k = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)
    n_loaded = 0
    for stem in view_names:
        p = sam3_cache / f"{stem}__{prompt_slug}.png"
        if p.is_file():
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                if k is not None:
                    m = cv2.dilate(m, k, iterations=1)
                masks[stem] = m
                n_loaded += 1
    print(f"    loaded {n_loaded}/{len(view_names)} SAM3 masks for prompt "
          f"`{prompt_slug}` from {sam3_cache} (dilate={dilate_px}px)")
    return masks


def _sample_color_for_points(view_source, best_view: np.ndarray,
                             best_uv: np.ndarray) -> np.ndarray:
    """Sample RGB (uint8 -> float [0,1]) from each point's best view photo
    via bilinear interpolation. Returns (N, 3) in [0,1]; points with
    best_view == -1 return NaN for downstream NN-copy.
    """
    import cv2
    views = view_source.all_views()
    n_pts = len(best_view)
    out = np.full((n_pts, 3), np.nan, dtype=np.float32)
    # Cache loaded images per view (lazy)
    cache = {}
    for i in range(n_pts):
        vi = int(best_view[i])
        if vi < 0:
            continue
        if vi not in cache:
            img = cv2.imread(str(views[vi].image_path), cv2.IMREAD_COLOR)
            cache[vi] = img[:, :, ::-1]  # BGR -> RGB
        img = cache[vi]
        u, v = best_uv[i]
        # Bilinear
        x0 = int(np.floor(u))
        y0 = int(np.floor(v))
        x1, y1 = x0 + 1, y0 + 1
        H, W = img.shape[:2]
        x0c = max(0, min(W - 1, x0))
        y0c = max(0, min(H - 1, y0))
        x1c = max(0, min(W - 1, x1))
        y1c = max(0, min(H - 1, y1))
        fx = u - x0
        fy = v - y0
        c00 = img[y0c, x0c].astype(np.float32)
        c01 = img[y0c, x1c].astype(np.float32)
        c10 = img[y1c, x0c].astype(np.float32)
        c11 = img[y1c, x1c].astype(np.float32)
        c = (c00 * (1 - fx) * (1 - fy) + c01 * fx * (1 - fy)
             + c10 * (1 - fx) * fy + c11 * fx * fy)
        out[i] = c / 255.0
    return out


def _nn_fill_missing(points_metric: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """For NaN rows in `rgb`, copy from nearest non-NaN neighbour (XZ Euclidean
    on the desk plane)."""
    has_color = ~np.isnan(rgb).any(axis=-1)
    if has_color.all():
        return rgb
    from scipy.spatial import cKDTree
    pts_xz = points_metric[:, [0, 2]]
    seen_pts = pts_xz[has_color]
    seen_rgb = rgb[has_color]
    tree = cKDTree(seen_pts)
    missing = ~has_color
    _, nn_idx = tree.query(pts_xz[missing], k=1)
    filled = rgb.copy()
    filled[missing] = seen_rgb[nn_idx]
    n_filled = int(missing.sum())
    print(f"  NN-copy: filled {n_filled} truly-unseen grid points "
          f"from {len(seen_pts)} observed")
    return filled


def _build_patch_ply(positions_splat: np.ndarray, rgb01: np.ndarray,
                     plane_quat: np.ndarray, scale_xz_log: float,
                     scale_normal_log: float, opacity_logit: float,
                     out_path: Path):
    n = len(positions_splat)
    arrs = []
    arrs.append(positions_splat[:, 0])                  # x
    arrs.append(positions_splat[:, 1])                  # y
    arrs.append(positions_splat[:, 2])                  # z
    arrs.append(np.zeros(n, dtype=np.float32))          # nx
    arrs.append(np.zeros(n, dtype=np.float32))          # ny
    arrs.append(np.zeros(n, dtype=np.float32))          # nz
    f_dc = ((rgb01 - 0.5) / SH_C0).astype(np.float32)
    arrs.append(f_dc[:, 0])
    arrs.append(f_dc[:, 1])
    arrs.append(f_dc[:, 2])
    for _ in range(45):
        arrs.append(np.zeros(n, dtype=np.float32))      # f_rest_*
    arrs.append(np.full(n, opacity_logit, dtype=np.float32))   # opacity
    arrs.append(np.full(n, scale_xz_log, dtype=np.float32))    # scale_0 (X tangent)
    arrs.append(np.full(n, scale_normal_log, dtype=np.float32)) # scale_1 (Y normal, thin)
    arrs.append(np.full(n, scale_xz_log, dtype=np.float32))    # scale_2 (Z tangent)
    arrs.append(np.full(n, plane_quat[0], dtype=np.float32))   # rot_w
    arrs.append(np.full(n, plane_quat[1], dtype=np.float32))   # rot_x
    arrs.append(np.full(n, plane_quat[2], dtype=np.float32))   # rot_y
    arrs.append(np.full(n, plane_quat[3], dtype=np.float32))   # rot_z

    dtype = [(name, "f4") for name in PLY_FIELDS]
    rows = np.empty(n, dtype=dtype)
    for name, col in zip(PLY_FIELDS, arrs):
        rows[name] = col
    el = PlyElement.describe(rows, "vertex")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([el]).write(str(out_path))


# ---------------------------------------------------------------------------
# Clone-mode helpers (K-NN Gaussian clone from surrounding scene splat)
# ---------------------------------------------------------------------------

def _splat_to_metric(positions_splat: np.ndarray,
                     R: np.ndarray, t: np.ndarray,
                     dp_scale: float, colmap_to_metric: float) -> np.ndarray:
    """Inverse of `_metric_to_splat`: splat -> metric metres.

    Forward chain: splat = dp_scale * (R @ (metric/colmap_to_metric) + t)
    Inverse:      metric = colmap_to_metric * R.T @ (splat/dp_scale - t)
    """
    colmap_pos = (R.T @ (positions_splat / dp_scale - t).T).T
    return (colmap_pos * colmap_to_metric).astype(np.float64)


def _load_scene_splat_metric(scene_splat_path: Path,
                             R: np.ndarray, t: np.ndarray,
                             dp_scale: float, colmap_to_metric: float):
    """Load the trained scene splat PLY, return:
      scene_arr:         (N, F) float32 rows (all properties, in PLY order)
      name_to_col:       {property_name: col_index}
      centres_metric:    (N, 3) float64 Gaussian centres in metric metres
      header_template:   str, a PLY header that matches this schema (used
                         to write the patch back in identical dtype)
    """
    pd = PlyData.read(str(scene_splat_path))
    v = pd["vertex"]
    prop_names = [p.name for p in v.properties]
    # sanity: SH degree 3 -> 45 f_rest_*
    n_rest = sum(1 for p in prop_names if p.startswith("f_rest_"))
    if n_rest != 45:
        raise RuntimeError(
            f"scene splat at {scene_splat_path} has {n_rest} f_rest_* props; "
            f"expected 45 (SH degree 3). Patch PLY_FIELDS is hard-coded to 45.")
    if prop_names != list(PLY_FIELDS):
        # Not fatal — as long as they're the same set. We use name_to_col.
        missing = set(PLY_FIELDS) - set(prop_names)
        extra = set(prop_names) - set(PLY_FIELDS)
        if missing or extra:
            raise RuntimeError(
                f"scene splat property set mismatch: missing={missing}, "
                f"extra={extra}")
    n = len(v.data)
    F = len(prop_names)
    scene_arr = np.empty((n, F), dtype=np.float32)
    name_to_col = {}
    for c, name in enumerate(prop_names):
        scene_arr[:, c] = np.asarray(v.data[name], dtype=np.float32)
        name_to_col[name] = c
    pos_splat = np.stack(
        [v.data["x"], v.data["y"], v.data["z"]], axis=-1).astype(np.float64)
    centres_metric = _splat_to_metric(
        pos_splat, R, t, dp_scale, colmap_to_metric)
    return scene_arr, name_to_col, centres_metric, prop_names


def _select_donor_mask(scene_centres_metric: np.ndarray,
                       object_mesh_paths: list,
                       inner_m: float,
                       outer_m: float,
                       y_ceiling_offset_m: float) -> np.ndarray:
    """Compute the donor mask over scene Gaussians.

    A Gaussian is a donor iff its metric centre:
      - is > inner_m from EVERY object mesh (i.e. not inside the crop volume)
      - is < outer_m from AT LEAST ONE object mesh (within the shell)
      - has Y <= (min over object meshes of y_min) + y_ceiling_offset_m
    """
    import open3d as o3d
    import trimesh as _tm
    n = len(scene_centres_metric)
    if n == 0:
        return np.zeros(0, dtype=bool)
    pts32 = scene_centres_metric.astype(np.float32)
    pts_t = o3d.core.Tensor(pts32)

    d_min_all = np.full(n, np.inf, dtype=np.float32)
    y_ceilings = []
    for mp in object_mesh_paths:
        m = _tm.load(str(mp), force="mesh", process=False)
        verts = np.asarray(m.vertices, dtype=np.float32)
        faces = np.asarray(m.faces, dtype=np.uint32)
        tm = o3d.t.geometry.TriangleMesh()
        tm.vertex.positions = o3d.core.Tensor(verts)
        tm.triangle.indices = o3d.core.Tensor(faces)
        rs = o3d.t.geometry.RaycastingScene()
        rs.add_triangles(tm)
        d = rs.compute_distance(pts_t).numpy()
        d_min_all = np.minimum(d_min_all, d)
        y_ceilings.append(float(verts[:, 1].min()) + float(y_ceiling_offset_m))

    outside_inner = d_min_all > inner_m
    inside_outer = d_min_all < outer_m
    # Y ceiling: keep only donors at or below the highest object's floor + offset
    y_ceiling = max(y_ceilings) if y_ceilings else np.inf
    below_ceiling = scene_centres_metric[:, 1] <= y_ceiling
    return outside_inner & inside_outer & below_ceiling


def _clone_gaussian_from_donors(pts_metric_grid: np.ndarray,
                                pts_splat_grid: np.ndarray,
                                scene_arr: np.ndarray,
                                scene_name_to_idx: dict,
                                donor_mask: np.ndarray,
                                donor_centres_metric: np.ndarray,
                                k: int = 8,
                                rot_mode: str = "nearest") -> np.ndarray:
    """K-NN clone-stamp Gaussian properties from `scene_arr` donors onto each
    grid point. Returns (P, F) rows in the same column order as scene_arr.

    Positions (x, y, z) come from `pts_splat_grid`; normals are zeroed.
    Colours (f_dc, f_rest), scales, opacity: per-channel MEDIAN of K donors
    (median in log/logit space where applicable). Rotation: nearest donor
    (rot_mode='nearest') or sign-corrected quaternion average ('avg').
    """
    from scipy.spatial import cKDTree
    donor_idx_all = np.where(donor_mask)[0]
    if donor_idx_all.size == 0:
        raise RuntimeError(
            "No donor Gaussians in shell; widen --donor-outer-m or check "
            "--object-mesh-dir / --scene-splat inputs.")
    tree = cKDTree(donor_centres_metric[donor_idx_all])

    # Column groups
    rot_cols = [scene_name_to_idx[f"rot_{i}"] for i in range(4)]
    sh_cols = ([scene_name_to_idx[f"f_dc_{i}"] for i in range(3)]
               + [scene_name_to_idx[f"f_rest_{i}"] for i in range(45)])
    scale_cols = [scene_name_to_idx[f"scale_{i}"] for i in range(3)]
    op_col = scene_name_to_idx["opacity"]
    x_col = scene_name_to_idx["x"]
    y_col = scene_name_to_idx["y"]
    z_col = scene_name_to_idx["z"]

    P = pts_metric_grid.shape[0]
    F = scene_arr.shape[1]
    out = np.zeros((P, F), dtype=np.float32)

    # Positions in splat space
    out[:, x_col] = pts_splat_grid[:, 0].astype(np.float32)
    out[:, y_col] = pts_splat_grid[:, 1].astype(np.float32)
    out[:, z_col] = pts_splat_grid[:, 2].astype(np.float32)
    # Normals stay at 0 (nx, ny, nz)

    K_eff = min(k, donor_idx_all.size)
    dists, nn_local = tree.query(pts_metric_grid, k=K_eff)
    if K_eff == 1:
        nn_local = nn_local[:, None]
        dists = dists[:, None]
    nn = donor_idx_all[nn_local]  # (P, K) global indices

    # SH DC + rest: per-channel median across K
    sh_stack = scene_arr[nn][:, :, sh_cols]           # (P, K, 48)
    out[:, sh_cols] = np.median(sh_stack, axis=1).astype(np.float32)

    # Scale: per-axis median in log space
    sc_stack = scene_arr[nn][:, :, scale_cols]        # (P, K, 3)
    out[:, scale_cols] = np.median(sc_stack, axis=1).astype(np.float32)

    # Opacity: median in logit space
    op_stack = scene_arr[nn][:, :, op_col]            # (P, K)
    out[:, op_col] = np.median(op_stack, axis=1).astype(np.float32)

    # Rotation
    if rot_mode == "nearest":
        out[:, rot_cols] = scene_arr[nn[:, 0]][:, rot_cols]
    else:  # 'avg' — sign-corrected quaternion mean
        qs = scene_arr[nn][:, :, rot_cols]                        # (P, K, 4)
        ref = qs[:, 0:1, :]
        dots = np.sum(qs * ref, axis=-1, keepdims=True)
        signs = np.where(dots >= 0, 1.0, -1.0).astype(np.float32)
        qs = qs * signs
        m = qs.mean(axis=1)
        m /= (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
        out[:, rot_cols] = m.astype(np.float32)

    # Diagnostic: high SH-DC spread flag
    if K_eff >= 2:
        max_spread = (sh_stack[:, :, :3].max(axis=1)
                      - sh_stack[:, :, :3].min(axis=1)).max(axis=1)
        n_noisy = int((max_spread > 0.15).sum())
        if n_noisy:
            print(f"    WARN {n_noisy}/{P} grid points sit at a colour "
                  f"boundary (K-neighbour SH-DC spread > 0.15); median used")
    return out


def _build_clone_patch_ply(rows: np.ndarray,
                           prop_names: list,
                           out_path: Path) -> None:
    """Write a structured PLY where the vertex element has one column per
    entry in `prop_names`, filled from `rows` (P x F float32) in column order.
    """
    P, F = rows.shape
    assert F == len(prop_names), (F, len(prop_names))
    dtype = [(name, "f4") for name in prop_names]
    out_rows = np.empty(P, dtype=dtype)
    for c, name in enumerate(prop_names):
        out_rows[name] = rows[:, c]
    el = PlyElement.describe(out_rows, "vertex")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([el]).write(str(out_path))


def process_object(slug: str, npz_path: Path, view_source,
                   scene_mesh_path: Path, dp_R, dp_t, dp_scale,
                   colmap_to_metric: float,
                   patch_radius_m: float, normal_radius_m: float,
                   opacity: float, out_dir: Path,
                   sam3_cache: Path | None = None,
                   sam3_dilate_px: int = 8,
                   topk_views: int = 5,
                   *,
                   mode: str = "clone",
                   scene_splat_path: Path | None = None,
                   object_mesh_dir: Path | None = None,
                   all_object_slugs: list | None = None,
                   donor_inner_m: float = 0.05,
                   donor_outer_m: float = 0.30,
                   donor_y_ceiling_m: float = 0.05,
                   donor_k: int = 200,
                   rot_mode: str = "nearest"):
    print(f"\n=== {slug}  [mode={mode}] ===")
    if not npz_path.exists():
        print(f"  missing {npz_path}; run _unseen_core_map.py first")
        return None
    data = np.load(npz_path)
    pts_m = data["grid_points"].astype(np.float64)
    seen = data["seen_count"]
    n_total = len(pts_m)
    n_unseen = int((seen == 0).sum())
    print(f"  grid: {n_total} points, {n_unseen} truly-unseen "
          f"({100 * n_unseen / n_total:.1f}%)")

    # Read fitted desk normal from NPZ if --auto-desk was used. Otherwise
    # fall back to world +Y (the old assumption).
    if "desk_normal_metric" in data.files:
        desk_normal_metric = data["desk_normal_metric"].astype(np.float64)
        desk_normal_metric = desk_normal_metric / np.linalg.norm(desk_normal_metric)
        print(f"  desk normal METRIC (auto-fit from NPZ): "
              f"({desk_normal_metric[0]:+.3f}, "
              f"{desk_normal_metric[1]:+.3f}, "
              f"{desk_normal_metric[2]:+.3f})  "
              f"tilt={np.degrees(np.arccos(desk_normal_metric[1])):.1f}°")
    else:
        desk_normal_metric = np.array([0.0, 1.0, 0.0])
        print(f"  desk normal METRIC: world +Y (no auto-fit in NPZ)")

    # Common: transform grid points metric -> splat space
    pts_s = _metric_to_splat(pts_m, dp_R, dp_t, dp_scale, colmap_to_metric)

    # ================================================================
    # CLONE MODE — K-NN Gaussian clone from the surrounding scene splat.
    # ================================================================
    if mode == "clone":
        if scene_splat_path is None or not Path(scene_splat_path).exists():
            raise RuntimeError(
                f"clone mode needs --scene-splat; got {scene_splat_path}")
        if object_mesh_dir is None:
            raise RuntimeError(
                "clone mode needs --object-mesh-dir")
        slug_list = list(all_object_slugs) if all_object_slugs else [slug]
        object_mesh_paths = [
            Path(object_mesh_dir) / s / f"{s}_extracted.ply"
            for s in slug_list
        ]
        missing = [p for p in object_mesh_paths if not p.exists()]
        if missing:
            raise RuntimeError(
                f"missing object mesh(es) for clone mode: {missing}")

        print(f"  clone mode: loading scene splat {scene_splat_path}")
        scene_arr, name_to_col, scene_centres_m, prop_names = (
            _load_scene_splat_metric(Path(scene_splat_path),
                                     dp_R, dp_t, dp_scale, colmap_to_metric))
        print(f"    scene splat: {scene_arr.shape[0]:,} gaussians, "
              f"{scene_arr.shape[1]} props")

        print(f"    computing donor mask (inner={donor_inner_m*100:.1f} cm, "
              f"outer={donor_outer_m*100:.1f} cm, y_ceil={donor_y_ceiling_m*100:.1f} cm) "
              f"against {len(object_mesh_paths)} object meshes...")
        donor_mask = _select_donor_mask(
            scene_centres_m, object_mesh_paths,
            inner_m=donor_inner_m,
            outer_m=donor_outer_m,
            y_ceiling_offset_m=donor_y_ceiling_m)
        n_donors = int(donor_mask.sum())
        print(f"    donors: {n_donors:,}/{scene_arr.shape[0]:,}")

        if n_donors < donor_k:
            print(f"    WARN <K donors ({n_donors} < {donor_k}); "
                  f"widening --donor-outer-m to 2*={donor_outer_m*2:.2f} m")
            donor_mask = _select_donor_mask(
                scene_centres_m, object_mesh_paths,
                inner_m=donor_inner_m,
                outer_m=donor_outer_m * 2.0,
                y_ceiling_offset_m=donor_y_ceiling_m)
            n_donors = int(donor_mask.sum())
            print(f"    donors after widen: {n_donors:,}")

        print(f"    K-NN clone (K={donor_k}, rot_mode={rot_mode}) "
              f"for {n_total} grid points ...")
        rows = _clone_gaussian_from_donors(
            pts_metric_grid=pts_m,
            pts_splat_grid=pts_s,
            scene_arr=scene_arr,
            scene_name_to_idx=name_to_col,
            donor_mask=donor_mask,
            donor_centres_metric=scene_centres_m,
            k=donor_k,
            rot_mode=rot_mode,
        )
        out_path = out_dir / f"desk_patch_{slug}.ply"
        _build_clone_patch_ply(rows, prop_names, out_path)
        sz = out_path.stat().st_size / 1024
        print(f"  wrote {out_path}  ({n_total} gaussians, {sz:.1f} KB) "
              f"[cloned from {n_donors:,} donors]")
        return out_path

    # ================================================================
    # VIEWS MODE — legacy top-K view-median SH-DC seed (kept as fallback).
    # ================================================================
    sam3_masks = None
    if sam3_cache is not None:
        view_names = [v.name for v in view_source.all_views()]
        sam3_masks = _load_sam3_masks(sam3_cache, slug, view_names,
                                      dilate_px=sam3_dilate_px)
        if not sam3_masks:
            sam3_masks = None
    print(f"  scoring top-{topk_views} views per point ...")
    topk_view, topk_uv = _topk_views_per_point(
        view_source, scene_mesh_path, pts_m.astype(np.float32), seen,
        sam3_masks=sam3_masks,
        desk_normal_metric=desk_normal_metric,
        k=topk_views)

    print(f"  median-sampling RGB from top-{topk_views} views ...")
    rgb01 = _sample_color_topk(view_source, topk_view, topk_uv)
    n_have_color = int((~np.isnan(rgb01).any(axis=-1)).sum())
    print(f"  sampled colour for {n_have_color}/{n_total}")

    rgb01 = _nn_fill_missing(pts_m, rgb01)

    # Splat-space desk normal: apply R only (no translation, no scale).
    # Uses the fitted (or fallback) metric normal from above.
    desk_normal_splat = dp_R @ desk_normal_metric
    desk_normal_splat = desk_normal_splat / np.linalg.norm(desk_normal_splat)
    plane_quat = _quat_y_to(desk_normal_splat)
    print(f"  desk normal in splat space: {desk_normal_splat.round(3)}")
    print(f"  quat (w,x,y,z): {plane_quat.round(3)}")

    # Scales in splat space. metric -> splat factor = dp_scale / colmap_to_metric
    # (since splat_dist = dp_scale * (R @ (metric/colmap_to_metric))).
    metric_to_splat_scalar = dp_scale / colmap_to_metric
    scale_xz_splat = patch_radius_m * metric_to_splat_scalar
    scale_normal_splat = normal_radius_m * metric_to_splat_scalar
    scale_xz_log = float(np.log(scale_xz_splat))
    scale_normal_log = float(np.log(scale_normal_splat))
    opacity_logit = float(np.log(opacity / (1 - opacity)))
    print(f"  scale: in-plane {patch_radius_m*1000:.1f} mm metric "
          f"-> log {scale_xz_log:.2f}; normal {normal_radius_m*1000:.1f} mm "
          f"-> log {scale_normal_log:.2f}; opacity logit {opacity_logit:.2f}")

    out_path = out_dir / f"desk_patch_{slug}.ply"
    _build_patch_ply(pts_s, rgb01.astype(np.float32),
                     plane_quat, scale_xz_log, scale_normal_log,
                     opacity_logit, out_path)
    sz = out_path.stat().st_size / 1024
    print(f"  wrote {out_path}  ({n_total} gaussians, {sz:.1f} KB)")
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project-root", type=Path, default=Path("."))
    p.add_argument("--scene-mesh", type=Path,
                   default=Path("output/mesh_v32_data3/mesh_v32_data3_openmvs.ply"))
    p.add_argument("--dataparser", type=Path,
                   default=Path("nerfstudio/dense/splatfacto/2026-06-07_203807/dataparser_transforms.json"))
    p.add_argument("--bounds-json", type=Path,
                   default=Path("colmap/dense/tof_bounds.json"))
    p.add_argument("--unseen-dir", type=Path,
                   default=Path("output/unseen_core_v32_data3"))
    p.add_argument("--prompts",
                   default="white_water_bottle,blue_box,red_lobster_figurine")
    p.add_argument("--out-dir", type=Path,
                   default=Path("output/desk_patch_v32_data3"))
    p.add_argument("--patch-radius-m", type=float, default=0.004,
                   help="In-plane Gaussian half-width (metric). 4 mm gives "
                        "small overlap with 5 mm grid spacing.")
    p.add_argument("--normal-radius-m", type=float, default=0.0008,
                   help="Normal-axis Gaussian half-width (metric). 0.8 mm = "
                        "thin flat-on-plane disk.")
    p.add_argument("--opacity", type=float, default=0.95)
    p.add_argument("--sam3-cache", type=Path,
                   default=Path("output/segmented_v32_data3/masks"),
                   help="Per-(view, prompt) SAM3 mask cache. Skips views where "
                        "the photo at the grid-point pixel shows the object, "
                        "so the patch colours stay pure desk (no red-lobster "
                        "bleed-through). Pass an empty string to disable.")
    p.add_argument("--sam3-dilate-px", type=int, default=8,
                   help="Morphological dilation of each SAM3 binary mask "
                        "(pixels) before using as a colour-sampler veto. "
                        "Catches anti-aliased object-edge contamination. "
                        "8 px ~ 4 mm on 1920x1080 / 1 m scene.")
    p.add_argument("--topk-views", type=int, default=5,
                   help="Use the K best-scoring views per grid point and "
                        "median their RGB samples. Robust to a single bleed-"
                        "through outlier. K=1 = old single-best-view behaviour.")

    # --- clone-mode options (mode == 'clone' is the new default) ---
    p.add_argument("--mode", choices=("clone", "views"), default="clone",
                   help="'clone' (default) K-NN clones Gaussians from the "
                        "surrounding scene splat; 'views' is the legacy "
                        "top-K view-median-RGB seed.")
    p.add_argument("--scene-splat", type=Path, default=None,
                   help="Clone-mode donor source; default "
                        "output/splat_<scene>_noinit_pruned.ply relative to "
                        "--project-root. Deduced from --scene-mesh stem "
                        "(prefix mesh_ -> splat_) when unset.")
    p.add_argument("--object-mesh-dir", type=Path, default=None,
                   help="Clone-mode object mesh root; default "
                        "output/segmented_<scene>_v9a_fp_v2.")
    p.add_argument("--donor-inner-m", type=float, default=0.05,
                   help="Clone-mode donor shell inner radius (metres).")
    p.add_argument("--donor-outer-m", type=float, default=0.30,
                   help="Clone-mode donor shell outer radius (metres).")
    p.add_argument("--donor-y-ceiling-m", type=float, default=0.30,
                   help="Clone-mode Y ceiling above the LOWEST object-mesh "
                        "y_min (metres).")
    p.add_argument("--donor-k", type=int, default=200,
                   help="Clone-mode K in K-NN.")
    p.add_argument("--rot-mode", choices=("nearest", "avg"),
                   default="nearest",
                   help="Clone-mode rotation strategy.")
    args = p.parse_args()
    if str(args.sam3_cache).strip() == "":
        args.sam3_cache = None

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[init] loading dataparser + bounds + view source")
    dp_R, dp_t, dp_scale = _load_dataparser(args.dataparser)
    colmap_to_metric = _load_colmap_to_metric(args.bounds_json)
    vs = V32ViewSource(args.project_root)
    print(f"[init] dp_scale={dp_scale:.5f}, colmap_to_metric=1/{1/colmap_to_metric:.4f}")
    print(f"[init] {len(vs)} views loaded")

    # Resolve clone-mode default paths from project_root + scene_mesh stem.
    if args.mode == "clone":
        scene_splat = args.scene_splat
        if scene_splat is None:
            # Try to derive: <root>/output/splat_<scene>_noinit_pruned.ply
            # where <scene> is the parent-directory name of scene-mesh.
            scene_dir = args.scene_mesh.parent.name  # e.g. mesh_v32_data3
            scene_name = (scene_dir[len("mesh_"):]
                          if scene_dir.startswith("mesh_") else scene_dir)
            scene_splat = (args.project_root / "output"
                           / f"splat_{scene_name}_noinit_pruned.ply")
            print(f"[init] auto scene-splat = {scene_splat}")
        else:
            scene_name = None
        object_mesh_dir = args.object_mesh_dir
        if object_mesh_dir is None:
            if scene_name is None:
                scene_dir = args.scene_mesh.parent.name
                scene_name = (scene_dir[len("mesh_"):]
                              if scene_dir.startswith("mesh_") else scene_dir)
            object_mesh_dir = (args.project_root / "output"
                               / f"segmented_{scene_name}_v9a_fp_v2")
            print(f"[init] auto object-mesh-dir = {object_mesh_dir}")
    else:
        scene_splat = None
        object_mesh_dir = None

    slug_list = [s.strip() for s in args.prompts.split(",") if s.strip()]
    for slug in slug_list:
        npz = args.unseen_dir / f"unseen_core_{slug}.npz"
        process_object(slug, npz, vs, args.scene_mesh,
                       dp_R, dp_t, dp_scale, colmap_to_metric,
                       args.patch_radius_m, args.normal_radius_m,
                       args.opacity, args.out_dir,
                       sam3_cache=args.sam3_cache,
                       sam3_dilate_px=args.sam3_dilate_px,
                       topk_views=args.topk_views,
                       mode=args.mode,
                       scene_splat_path=scene_splat,
                       object_mesh_dir=object_mesh_dir,
                       all_object_slugs=slug_list,
                       donor_inner_m=args.donor_inner_m,
                       donor_outer_m=args.donor_outer_m,
                       donor_y_ceiling_m=args.donor_y_ceiling_m,
                       donor_k=args.donor_k,
                       rot_mode=args.rot_mode)


if __name__ == "__main__":
    main()
