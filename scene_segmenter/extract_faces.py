"""Weighted multi-view face extraction.

Improvements over the original automated_trimming remove_faces:
  * Vertex projection via real intrinsics K and per-view w2c (OpenCV).
  * Per-pixel Z-buffer (occlusion-correct) -- same as the original.
  * **Weighted voting**: each view contributes a float weight per face equal
    to (view_angle_weight * pixel_coverage), not a binary vote.
      - view_angle_weight = max(0, dot(-face_normal, view_dir))
        head-on views weight 1.0, glancing views weight ~0.
      - pixel_coverage = fraction of the face's 3 vertices that are inside
        the mask AND closest to their pixel (0/3, 1/3, 2/3, 3/3).
  * Continuous score threshold instead of integer min_votes. Score is
    normalised by sum of theoretical weights (camera-facing views in which
    the face was visible), so it's a fraction in [0, 1].
  * **Connected-components filter**: after thresholding, keep components
    above a min-face-count fraction of the largest. Drops floater fragments
    that the binary voter would have kept.
  * **Two modes**: EXTRACT (keep matched faces) and REMOVE (keep unmatched).
  * Returns both the face mask AND the per-face score, so the caller can
    threshold differently.

The math is in metric METRES (same as the scale-calibrated mesh + the
metric-rescaled poses from views.V32ViewSource).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class FaceExtractionResult:
    prompt: str
    n_faces_total: int
    face_score: np.ndarray            # (F,) float in [0, 1]
    face_mask: np.ndarray             # (F,) bool, True if face is part of object
    n_components_before_cc: int
    n_components_after_cc: int
    n_views_contributing: int         # views that had a non-empty mask


def _project_samples_to_view(
    samples: np.ndarray,
    w2c: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    z_eps: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project Nx3 world-space points into the view's image plane.

    Returns: px_clip, py_clip (int32, clipped to image bounds), depth
    (float32, world units along camera +Z), valid (bool: in front of
    camera AND in image bounds).
    """
    N = len(samples)
    homog = np.hstack([samples, np.ones((N, 1))])
    p_cam = (w2c @ homog.T).T[:, :3]
    depth = p_cam[:, 2]
    in_front = depth > z_eps
    d_safe = np.where(in_front, depth, z_eps)
    pix_x = K[0, 0] * (p_cam[:, 0] / d_safe) + K[0, 2]
    pix_y = K[1, 1] * (p_cam[:, 1] / d_safe) + K[1, 2]
    px = np.round(pix_x).astype(np.int32)
    py = np.round(pix_y).astype(np.int32)
    in_bounds = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    valid = in_front & in_bounds
    return (np.clip(px, 0, width - 1),
            np.clip(py, 0, height - 1),
            depth.astype(np.float32),
            valid)


def _mesh_visible_vertices_from_view(
    vertices: np.ndarray,
    w2c: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    z_eps: float = 1e-4,
    z_buffer_tol: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project mesh vertices and return per-vertex visibility via a
    vertex-built z-buffer. Kept for backwards-compat; the dense
    7-sample path in accumulate_face_scores uses the lower-level
    _project_samples_to_view directly.
    """
    px, py, depth, valid = _project_samples_to_view(
        vertices, w2c, K, width, height, z_eps=z_eps,
    )
    z_buf = np.full(height * width, np.inf, dtype=np.float32)
    v_idx = np.where(valid)[0]
    if len(v_idx):
        flat_pix = py[v_idx] * width + px[v_idx]
        np.minimum.at(z_buf, flat_pix, depth[v_idx])
    z_buf = z_buf.reshape(height, width)
    z_at = z_buf[py, px]
    z_buf_hit = z_at < np.inf
    rel_diff = np.abs(depth - z_at)
    visible = valid & z_buf_hit & (rel_diff <= z_at * z_buffer_tol + 1e-3)
    return px, py, visible


def _build_raycasting_scene(mesh):
    """Build an Open3D RaycastingScene from a trimesh mesh. Returns the
    scene object plus an INVALID_ID sentinel for unhit pixels.

    The scene is built once per pipeline run and reused across all
    views. BVH build is ~1-2s for ~1M faces; ray-casting is the loop
    hot path.
    """
    import open3d as o3d
    import numpy as np
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    tmesh = o3d.t.geometry.TriangleMesh()
    tmesh.vertex.positions = o3d.core.Tensor(verts)
    tmesh.triangle.indices = o3d.core.Tensor(faces)
    scene = o3d.t.geometry.RaycastingScene()
    geom_id = scene.add_triangles(tmesh)
    return scene, int(o3d.t.geometry.RaycastingScene.INVALID_ID)


def _rasterize_view_face_ids(scene, invalid_id, view):
    """Per-pixel FaceID + depth via ray-casting. Returns:
        face_id : (H, W) int64, -1 where no triangle was hit
        depth   : (H, W) float32, +inf where no hit

    Why ray-casting instead of our old vertex-projection Z-buffer:
    each pixel's ray finds the actual closest triangle via BVH, so
    occlusion is exact. Background ghost faces no longer bleed through
    pixels of front-facing geometry just because that pixel didn't
    happen to have a vertex projected onto it.
    """
    import open3d as o3d
    import numpy as np
    K_f32 = o3d.core.Tensor(view.K.astype(np.float32))
    extr_f32 = o3d.core.Tensor(view.w2c.astype(np.float32))
    rays = scene.create_rays_pinhole(
        intrinsic_matrix=K_f32,
        extrinsic_matrix=extr_f32,
        width_px=int(view.width),
        height_px=int(view.height),
    )
    ans = scene.cast_rays(rays)
    face_id_u32 = ans["primitive_ids"].numpy()
    depth = ans["t_hit"].numpy()
    face_id = face_id_u32.astype(np.int64)
    face_id[face_id_u32 == invalid_id] = -1
    return face_id, depth


def _erode_dilate_mask(mask: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (eroded, dilated) versions of a binary mask using a
    structuring element with the given pixel radius.

    Used to define the 3-way evidence regions:
        positive = eroded               (confidently inside)
        negative = ~dilated             (confidently outside)
        uncertain = dilated & ~eroded   (SAM boundary band; vote-skip)
    """
    if radius <= 0:
        return mask.copy(), mask.copy()
    from PIL import Image as _PILImage, ImageFilter as _IF
    pil = _PILImage.fromarray((mask.astype(np.uint8) * 255))
    k = radius * 2 + 1
    eroded = np.array(pil.filter(_IF.MinFilter(k))) > 128
    dilated = np.array(pil.filter(_IF.MaxFilter(k))) > 128
    return eroded, dilated


def accumulate_face_scores_rasterized(
    mesh,
    views_info: list,
    masks_by_view: Dict[str, np.ndarray],
    *,
    min_pixel_count: int = 8,
    mask_margin_px: int = 3,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Consensus voting via per-pixel FaceID rasterization, with 3-way
    evidence accumulation (positive / negative / uncertain).

    Per view:
      1. Ray-cast the mesh to get per-pixel FaceID + depth maps.
         Occlusion is exact via BVH.
      2. Erode the SAM3 mask by `mask_margin_px` -> "positive" region
         (confidently inside the object).
      3. Dilate the SAM3 mask by `mask_margin_px` -> the complement of
         "negative" region (confidently outside).
      4. The band between erode and dilate is "uncertain" -- skip those
         pixels entirely (they can't bias the score either way).
      5. For each visible pixel: if positive, +1 to that face's positive
         count; if negative, +1 to negative count; if uncertain, skip.

    Final score per face:
        score = positive / (positive + negative + eps)

    Why this is much stronger than `in_mask / total`: a face that
    straddles the SAM3 silhouette boundary used to get dragged toward
    0.5 by the boundary pixels (some in, some out). With 3-way evidence
    those boundary pixels just don't vote. The face is judged only on
    confident signal -> interior swiss-cheese holes vanish.
    """
    F = len(mesh.faces)
    pos_count = np.zeros(F, dtype=np.int64)
    neg_count = np.zeros(F, dtype=np.int64)
    seen_count = np.zeros(F, dtype=np.int64)         # any visible pixel (for face_voted)

    if verbose:
        print(f"[rasterize] building BVH over {len(mesh.faces):,} faces ...")
        print(f"[rasterize] mask margin: {mask_margin_px} px (erode/dilate band ignored)")
    scene, invalid_id = _build_raycasting_scene(mesh)

    n_views_used = 0
    for vi, view in enumerate(views_info):
        if view.name not in masks_by_view:
            continue
        mask = masks_by_view[view.name]
        if mask is None or not mask.any():
            continue
        n_views_used += 1

        face_id, _ = _rasterize_view_face_ids(scene, invalid_id, view)

        # Resize mask to camera shape if needed
        mask_arr = mask.astype(bool)
        mh, mw = mask_arr.shape
        if (mh, mw) != (view.height, view.width):
            if abs(mh * mw - view.height * view.width) > 0.01 * view.height * view.width:
                raise ValueError(
                    f"[rasterize] view {view.name!r}: mask shape {mask_arr.shape} "
                    f"differs from camera {view.height}x{view.width} by more than 1%."
                )
            from PIL import Image as _PILImage
            mask_arr = np.array(
                _PILImage.fromarray((mask_arr.astype(np.uint8) * 255)).resize(
                    (view.width, view.height), _PILImage.NEAREST
                )
            ) > 0

        positive_mask, dilated = _erode_dilate_mask(mask_arr, mask_margin_px)
        negative_mask = ~dilated

        valid = face_id >= 0
        if not valid.any():
            continue
        flat_fid = face_id[valid]
        flat_pos = positive_mask[valid].astype(np.int32)
        flat_neg = negative_mask[valid].astype(np.int32)

        view_pos = np.bincount(flat_fid, weights=flat_pos, minlength=F).astype(np.int64)
        view_neg = np.bincount(flat_fid, weights=flat_neg, minlength=F).astype(np.int64)
        view_seen = np.bincount(flat_fid, minlength=F).astype(np.int64)

        pos_count += view_pos
        neg_count += view_neg
        seen_count += view_seen

        if verbose:
            n_face_pos = int((view_pos > 0).sum())
            n_face_neg = int((view_neg > 0).sum())
            print(f"  view {vi+1}/{len(views_info)} {view.name}  "
                  f"mask_cov={mask_arr.mean()*100:.1f}%  "
                  f"faces_pos={n_face_pos:,}  faces_neg={n_face_neg:,}")

    # Three-way evidence score
    eps = 1e-9
    enough = (pos_count + neg_count) >= min_pixel_count
    score = np.zeros(F, dtype=np.float64)
    score[enough] = pos_count[enough] / (pos_count[enough] + neg_count[enough] + eps)
    face_voted = seen_count > 0

    if verbose:
        n_enough = int(enough.sum())
        n_high = int((score >= 0.7).sum())
        print(f"[rasterize] {n_enough:,} faces with >= {min_pixel_count} confident pixel votes; "
              f"{n_high:,} scored >= 0.70")
    return score, face_voted, n_views_used


def _compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per-face area-weighted normals; returns unit-normalised (F, 3)."""
    p0 = vertices[faces[:, 0]]
    p1 = vertices[faces[:, 1]]
    p2 = vertices[faces[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)
    nn = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(nn, 1e-12)


def accumulate_face_scores(
    mesh,                               # trimesh.Trimesh
    views_info: list,                   # list of ViewInfo (from views.py)
    masks_by_view: Dict[str, np.ndarray],
    *,
    z_buffer_tol: float = 0.02,
    min_view_angle_cos: float = 0.10,
    samples_per_face: int = 7,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Run weighted voting across every view that has a mask.

    `samples_per_face` selects how each face is sampled against the mask:
        3 -- 3 vertices only (legacy; cov_frac steps in {0, 1/3, 2/3, 1})
        7 -- 3 vertices + 3 edge midpoints + 1 centroid (default; finer
             coverage gradient -> dramatically fewer holes from edge noise)

    Returns:
        face_score    : (F,) float in [0, 1]  -- normalised score per face
        face_voted    : (F,) bool             -- True if any view voted nonzero
        n_views_used  : int                   -- number of views that contributed
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    F = len(faces)
    f0, f1, f2 = faces[:, 0], faces[:, 1], faces[:, 2]

    face_normals = _compute_face_normals(vertices, faces)

    # ---- Precompute per-face extra sample points (world coords) ----
    use_dense = samples_per_face >= 7
    if use_dense:
        p0 = vertices[f0]
        p1 = vertices[f1]
        p2 = vertices[f2]
        m01 = (p0 + p1) * 0.5
        m12 = (p1 + p2) * 0.5
        m20 = (p2 + p0) * 0.5
        centroid = (p0 + p1 + p2) / 3.0
        # extras_per_face shape: (F, 4, 3). Flatten for projection.
        extras = np.stack([m01, m12, m20, centroid], axis=1)  # (F, 4, 3)
        extras_flat = extras.reshape(-1, 3)                   # (4F, 3)
        denom = 7.0
    else:
        denom = 3.0

    # Accumulators
    weighted_sum = np.zeros(F, dtype=np.float64)
    weight_total = np.zeros(F, dtype=np.float64)

    n_views_used = 0
    for vi, view in enumerate(views_info):
        if view.name not in masks_by_view:
            continue
        mask = masks_by_view[view.name]
        if mask is None or not mask.any():
            continue

        n_views_used += 1

        # ---- Project vertices and (if dense) face-interior extras ----
        pxv, pyv, dv, validv = _project_samples_to_view(
            vertices, view.w2c, view.K, view.width, view.height,
        )
        if use_dense:
            pxe, pye, de, valide = _project_samples_to_view(
                extras_flat, view.w2c, view.K, view.width, view.height,
            )

        # ---- Build a unified z-buffer from ALL projected samples ----
        # The vertex-only z-buffer from the legacy path was sparse (only
        # vertex pixels written). Adding the 4F extras per view writes
        # ~5x more pixels, making the occlusion test substantially less
        # noisy. We still cap at "min depth at this pixel" so the
        # semantics don't change.
        z_buf = np.full(view.height * view.width, np.inf, dtype=np.float32)
        v_idx = np.where(validv)[0]
        if len(v_idx):
            flat_pix = pyv[v_idx] * view.width + pxv[v_idx]
            np.minimum.at(z_buf, flat_pix, dv[v_idx])
        if use_dense:
            e_idx = np.where(valide)[0]
            if len(e_idx):
                flat_pix_e = pye[e_idx] * view.width + pxe[e_idx]
                np.minimum.at(z_buf, flat_pix_e, de[e_idx])
        z_buf = z_buf.reshape(view.height, view.width)

        # ---- Per-sample visibility against the unified z-buffer ----
        def _vis_against_zbuf(px, py, depth, valid):
            z_at = z_buf[py, px]
            return (valid
                    & (z_at < np.inf)
                    & (np.abs(depth - z_at) <= z_at * z_buffer_tol + 1e-3))

        vis_v = _vis_against_zbuf(pxv, pyv, dv, validv)
        vis_e = _vis_against_zbuf(pxe, pye, de, valide) if use_dense else None

        # ---- Resize mask to camera shape if needed ----
        mask_arr = mask.astype(bool)
        mh, mw = mask_arr.shape
        if (mh, mw) != (view.height, view.width):
            if abs(mh * mw - view.height * view.width) > 0.01 * view.height * view.width:
                raise ValueError(
                    f"[extract] view {view.name!r}: mask shape {mask_arr.shape} "
                    f"differs from camera {view.height}x{view.width} by more "
                    f"than 1% in area - probable rotation or wrong-frame mask."
                )
            from PIL import Image as _PILImage
            mask_arr = np.array(
                _PILImage.fromarray((mask_arr.astype(np.uint8) * 255)).resize(
                    (view.width, view.height), _PILImage.NEAREST
                )
            ) > 0

        # ---- Per-face coverage: sum in_mask across all 3 or 7 samples ----
        in_mask_v = vis_v & mask_arr[pyv, pxv]              # (V,)
        cov = (in_mask_v[f0].astype(np.int16) +
               in_mask_v[f1].astype(np.int16) +
               in_mask_v[f2].astype(np.int16))              # 0..3

        if use_dense:
            in_mask_e = vis_e & mask_arr[pye, pxe]          # (4F,)
            cov_e_per_face = in_mask_e.reshape(F, 4).sum(axis=1).astype(np.int16)
            cov = cov + cov_e_per_face                       # 0..7

        cov_frac = cov.astype(np.float64) / denom

        # face_visible_any: any of the per-face samples were visible
        face_visible_any = vis_v[f0] | vis_v[f1] | vis_v[f2]
        if use_dense:
            face_visible_any = face_visible_any | vis_e.reshape(F, 4).any(axis=1)

        # View-angle weight: how head-on is the camera relative to the face?
        centroids_w = (vertices[f0] + vertices[f1] + vertices[f2]) / 3.0
        view_dirs = centroids_w - view.camera_position[None, :]
        view_dirs /= np.linalg.norm(view_dirs, axis=1, keepdims=True) + 1e-12
        head_on = -np.einsum("ij,ij->i", face_normals, view_dirs)
        head_on = np.maximum(head_on, 0.0)

        eligible = face_visible_any & (head_on > min_view_angle_cos)
        contrib = head_on * cov_frac
        weighted_sum[eligible] += contrib[eligible]
        weight_total[eligible] += head_on[eligible]

        if verbose:
            print(f"  view {vi+1}/{len(views_info)} {view.name}  "
                  f"mask_cov={mask_arr.mean()*100:.1f}%  "
                  f"face_hits={(cov > 0).sum():,}")

    # Normalise: score = sum(angle * coverage) / sum(angle for eligible views).
    # Faces never seen by any eligible view: score stays 0.  np.where avoids
    # the divide-by-zero edge case cleanly (review nit L-1).
    eps = 1e-9
    face_score = np.where(weight_total > eps,
                          weighted_sum / np.maximum(weight_total, eps),
                          0.0)
    face_score = np.clip(face_score, 0.0, 1.0)

    face_voted = weighted_sum > 0.0
    return face_score, face_voted, n_views_used


def smooth_face_scores(
    mesh,                                 # trimesh.Trimesh
    scores: np.ndarray,                   # (F,) float
    iterations: int = 2,
    alpha: float = 0.5,
    verbose: bool = True,
) -> np.ndarray:
    """Laplacian-smooth per-face scores across mesh adjacency.

    Why this exists: the raw score has only a 3-sample-per-face cov_frac,
    so faces in the *interior* of an object get noisy scores (e.g. 0.4
    next to 0.18 next to 0.6). A hard threshold turns that noise into
    swiss-cheese holes. One or two rounds of `new = (1-a)*self + a*avg(
    neighbors)` averages each face with its edge-neighbors, pulling
    isolated low-scoring faces inside a high-scoring region above the
    threshold while leaving the silhouette boundary roughly in place.

    iterations=0 disables smoothing (current/legacy behaviour).
    """
    if iterations <= 0:
        return scores
    fa = np.asarray(mesh.face_adjacency)         # (E, 2) edge-neighbor face pairs
    if len(fa) == 0:
        return scores
    F = len(scores)
    smoothed = scores.astype(np.float64).copy()
    for k in range(iterations):
        neighbor_sum = np.zeros(F, dtype=np.float64)
        neighbor_count = np.zeros(F, dtype=np.int32)
        np.add.at(neighbor_sum, fa[:, 0], smoothed[fa[:, 1]])
        np.add.at(neighbor_sum, fa[:, 1], smoothed[fa[:, 0]])
        np.add.at(neighbor_count, fa[:, 0], 1)
        np.add.at(neighbor_count, fa[:, 1], 1)
        neighbor_avg = neighbor_sum / np.maximum(neighbor_count, 1)
        smoothed = (1.0 - alpha) * smoothed + alpha * neighbor_avg
    if verbose:
        print(f"  score smoothing: {iterations} iter alpha={alpha}")
    return smoothed


def dilate_face_mask(
    mesh,
    mask: np.ndarray,
    iterations: int = 1,
    min_true_neighbors: int = 1,
    verbose: bool = True,
) -> np.ndarray:
    """Morphological dilation on a face mask, using mesh edge adjacency.

    For each False face, flips it to True if it has at least
    `min_true_neighbors` True edge-neighbors. Iterate to grow further.

    Purpose: the raw thresholded mask is often a sprinkle of small
    disconnected pieces because the score has noise at the per-face
    level. Dilating by 1-3 rings closes 1-3 face-wide holes inside
    each region and fuses neighboring fragments into one piece.
    Silhouette grows slightly (1-3 faces outward).
    """
    if iterations <= 0:
        return mask
    fa = np.asarray(mesh.face_adjacency)
    if len(fa) == 0:
        return mask
    F = len(mask)
    m = mask.copy()
    for k in range(iterations):
        neighbor_true = np.zeros(F, dtype=np.int32)
        np.add.at(neighbor_true, fa[:, 0], m[fa[:, 1]].astype(np.int32))
        np.add.at(neighbor_true, fa[:, 1], m[fa[:, 0]].astype(np.int32))
        new_true = (~m) & (neighbor_true >= min_true_neighbors)
        if verbose:
            print(f"  dilate iter {k+1}: +{int(new_true.sum()):,} faces "
                  f"(min_true_neighbors={min_true_neighbors})")
        m = m | new_true
    return m


def filter_by_components(
    mesh,                       # trimesh.Trimesh
    selected: np.ndarray,       # (F,) bool
    min_component_fraction: float = 0.05,
    verbose: bool = True,
) -> tuple[np.ndarray, int, int]:
    """Keep only large connected components within the selected face set.

    Returns:
        keep_mask          : (F,) bool, filtered selection
        n_components_before: int
        n_components_after : int
    """
    import trimesh

    if not selected.any():
        return selected.copy(), 0, 0

    # Build the sub-mesh of selected faces only
    sub_faces = np.where(selected)[0]
    fa = mesh.face_adjacency       # (E, 2) pairs of adjacent face indices
    # Sub-adjacency: only edges where both faces are selected
    sel_set = set(sub_faces.tolist())
    sub_fa_mask = np.array([(fa[i, 0] in sel_set) and (fa[i, 1] in sel_set)
                            for i in range(len(fa))], dtype=bool)
    sub_fa = fa[sub_fa_mask]

    components = trimesh.graph.connected_components(
        sub_fa, nodes=sub_faces, min_len=1,
    )
    n_before = len(components)
    if n_before == 0:
        return selected.copy(), 0, 0

    sizes = np.array([len(c) for c in components])
    largest = int(sizes.max())
    # Threshold floor of 1 so a single-component result always keeps the
    # largest component, even if min_component_fraction rounds to zero
    # (review bug H-4).
    threshold = max(1, int(largest * min_component_fraction))
    keep_components = [c for c, s in zip(components, sizes) if s >= threshold]
    n_after = len(keep_components)

    keep_face_idx = (np.concatenate(keep_components)
                     if keep_components else np.array([], dtype=np.int64))
    keep_mask = np.zeros(len(mesh.faces), dtype=bool)
    if len(keep_face_idx):
        keep_mask[keep_face_idx] = True

    if verbose:
        print(f"  CC filter: {n_before} -> {n_after} components  "
              f"(largest {largest:,} faces, kept any >= {int(largest * min_component_fraction):,})")

    return keep_mask, n_before, n_after


def extract_object(
    mesh,
    views_info: list,
    masks_by_view: Dict[str, np.ndarray],
    *,
    prompt: str,
    score_threshold: float = 0.40,
    min_component_fraction: float = 0.05,
    z_buffer_tol: float = 0.02,
    min_view_angle_cos: float = 0.10,
    smooth_iterations: int = 2,
    smooth_alpha: float = 0.5,
    dilate_iterations: int = 0,
    dilate_min_neighbors: int = 1,
    samples_per_face: int = 7,
    use_rasterizer: bool = True,
    min_pixel_count: int = 8,
    mask_margin_px: int = 3,
    apply_cc_filter: bool = True,
    verbose: bool = True,
) -> FaceExtractionResult:
    """High-level: score every face, threshold, optionally CC-filter.

    The returned `face_mask` is True for faces belonging to the named object.
    To EXTRACT: keep faces where face_mask is True.
    To REMOVE:  keep faces where face_mask is False.
    """
    if verbose:
        print(f"\n[extract] prompt={prompt!r}  votes from {len(masks_by_view)} views ...")

    if use_rasterizer:
        score, voted, n_used = accumulate_face_scores_rasterized(
            mesh, views_info, masks_by_view,
            min_pixel_count=min_pixel_count,
            mask_margin_px=mask_margin_px,
            verbose=verbose,
        )
    else:
        score, voted, n_used = accumulate_face_scores(
            mesh, views_info, masks_by_view,
            z_buffer_tol=z_buffer_tol,
            min_view_angle_cos=min_view_angle_cos,
            samples_per_face=samples_per_face,
            verbose=verbose,
        )

    if smooth_iterations > 0:
        score = smooth_face_scores(
            mesh, score,
            iterations=smooth_iterations,
            alpha=smooth_alpha,
            verbose=verbose,
        )

    selected = score >= score_threshold
    if verbose:
        print(f"  faces scored above {score_threshold}: {selected.sum():,} / {len(score):,}  "
              f"({100*selected.sum()/len(score):.1f}%)")

    if dilate_iterations > 0 and selected.any():
        n_before_dilate = int(selected.sum())
        selected = dilate_face_mask(
            mesh, selected,
            iterations=dilate_iterations,
            min_true_neighbors=dilate_min_neighbors,
            verbose=verbose,
        )
        if verbose:
            print(f"  after dilate: {selected.sum():,} faces "
                  f"(+{int(selected.sum()) - n_before_dilate:,})")

    n_before_cc, n_after_cc = 0, 0
    if apply_cc_filter and selected.any():
        selected, n_before_cc, n_after_cc = filter_by_components(
            mesh, selected, min_component_fraction=min_component_fraction,
            verbose=verbose,
        )
        if verbose:
            print(f"  after CC filter: {selected.sum():,} faces "
                  f"({100*selected.sum()/len(score):.1f}%)")

    return FaceExtractionResult(
        prompt=prompt,
        n_faces_total=len(mesh.faces),
        face_score=score,
        face_mask=selected,
        n_components_before_cc=n_before_cc,
        n_components_after_cc=n_after_cc,
        n_views_contributing=n_used,
    )
