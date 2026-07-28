"""Agent B -- surgical repair actions for the iterative refinement loop.

Each function takes a PLY on disk, applies one focused filter, and
writes a new PLY (or overwrites in place). The PLY schema is
preserved exactly (structured ndarray via plyfile).

Actions:
    drop_residue_gaussians           -- drop by SAM3 residue score gated by
                                        Y-above-plane OR tightened colour
                                        test (OR across slugs)
    drop_subplane_gaussians          -- drop centres below the fitted desk plane
                                        inside an object's padded AABB
    drop_needle_gaussians_in_region  -- drop long anisotropic ellipsoids
                                        inside an object's padded AABB
    drop_residue_mesh_verts          -- mesh: drop faces where any vertex is
                                        residue-flagged (preserves vertex colours)
    densify_patch_gaussians          -- clone nearest existing Gaussian per
                                        sparse desk-patch cell (donor colour +
                                        rotation, overwritten opacity/scale)
    adjust_crop_dist_for_object      -- config-only: bump per-slug crop_dist_m

Every function that mutates a PLY snapshots the input (via copy2) into
`snapshot_dir` first, so Agent C can roll back a step by restoring the
snapshot.

Patch protection (2026-07-20):
    build_patch_protection_set(...) returns a metric (M, 3) array
    of desk-patch Gaussian positions loaded from the input
    desk_patch_<slug>.ply files. Every drop_* function accepts an
    optional patch_positions ndarray; any Gaussian whose METRIC
    centre is within 1 mm of a patch position is *never* dropped,
    even if it independently satisfies the drop criterion. This
    guards against v1 where our own patch fills were shredded by
    the residue detector because they projected into the SAM3
    monitor / object masks.

Colour reference filter (2026-07-20):
    compute_color_references(...) returns
        {slug: {"obj_rgb": (r,g,b), "desk_rgb": (r,g,b)}} in [0,1].
    obj_rgb   = median vertex colour of the extracted object mesh.
    desk_rgb  = median vertex colour of scene-mesh verts near the
                fitted desk plane, 15-30 cm outside the object AABB.
    drop_residue_gaussians only drops a Gaussian whose f_dc SH
    colour is CLOSER to the object colour than to the desk colour
    (Euclidean in linear [0,1] RGB). This stops the tan desk-patch
    fill from being removed just because it happens to project
    into a SAM3 mask.

Y-above-plane drop (2026-07-20, v3 tightening):
    drop_residue_gaussians additionally accepts `desk_planes` and
    `above_plane_drop_m` (default 3 cm). Any residue-flagged Gaussian
    whose METRIC centre lies more than `above_plane_drop_m` above the
    fitted desk plane (signed distance with up-normal) is DROPPED
    regardless of colour -- desks are planar, so anything floating
    above the surface is residue by construction. Patch protection
    still wins (patch-position Gaussians are never dropped).

    The colour filter is simultaneously tightened via `color_margin`
    (default 0.85): a Gaussian is now considered "colour-drops" only
    when d_obj < color_margin * d_desk (i.e. CLEARLY closer to obj),
    which rejects the ambiguous middle-ground colours the wider
    d_obj < d_desk test used to keep.

    Combined per-slug drop rule when either plane or colour info is
    available:
        drop = residue_hit & (above_plane | desk_closer)
    If neither is available for a slug we fall back to the v1
    behaviour of dropping every residue_hit for that slug.

Reused helpers:
    _pipeline_full._read_ply / _write_ply           (structured-ndarray)
    _spatial_crop_splat._splat_to_metric
    _spatial_crop_splat._load_dataparser_transform
"""
from __future__ import annotations

import gc
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

from _pipeline_full import _read_ply, _write_ply  # noqa: E402
from _spatial_crop_splat import (  # noqa: E402
    _load_dataparser_transform,
    _splat_to_metric,
)

SH_C0 = 0.28209479177387814


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(msg, flush=True)


def _snapshot(src: Path, snapshot_dir: Optional[Path], tag: str) -> Optional[Path]:
    if snapshot_dir is None:
        return None
    src = Path(src)
    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    dst = snapshot_dir / f"{src.stem}__{tag}{src.suffix}"
    if not src.exists():
        return None
    shutil.copy2(src, dst)
    return dst


def _atomic_write_ply(rows: np.ndarray, dst: Path) -> None:
    """Write PLY atomically (temp file + os.replace) so partial crashes
    never leave a half-written file on the working path.

    Windows note: plyfile.PlyData.read defaults to mmap=True, so any
    earlier reader (residue detection or a prior _load_splat_rows call
    in the same iteration) may still hold a memory-map on `dst`. That
    turns os.replace into WinError 5. We force a GC before replacing,
    and fall back to unlink+rename if the mmap still lingers.
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    _write_ply(rows, tmp)
    gc.collect()
    try:
        os.replace(tmp, dst)
    except PermissionError:
        gc.collect()
        try:
            os.replace(tmp, dst)
        except PermissionError:
            if dst.exists():
                try:
                    dst.unlink()
                except PermissionError:
                    gc.collect()
                    dst.unlink()
            os.rename(tmp, dst)


def _load_splat_rows(splat_path: Path):
    """Return (rows_struct, dtype, n_rows) via _pipeline_full._read_ply.

    Copies the structured array out of any plyfile mmap so the source
    file handle can release. Without this, Windows fails a later
    os.replace on the same path with WinError 5.
    """
    rows_mm, dtype = _read_ply(Path(splat_path))
    rows = np.array(rows_mm, dtype=dtype, copy=True)
    del rows_mm
    gc.collect()
    return rows, dtype, int(len(rows))


def _splat_positions_metric(rows: np.ndarray, *,
                            dp_R: np.ndarray, dp_t: np.ndarray,
                            dp_scale: float,
                            colmap_to_metric: float) -> np.ndarray:
    """(N, 3) float64 metric positions from splat-space rows."""
    xyz_splat = np.stack([np.asarray(rows["x"], dtype=np.float64),
                          np.asarray(rows["y"], dtype=np.float64),
                          np.asarray(rows["z"], dtype=np.float64)], axis=-1)
    return _splat_to_metric(xyz_splat, dp_R, dp_t, dp_scale, colmap_to_metric)


def _bulk_or_scores(residue_scores: dict, n_rows: int,
                    slugs, threshold: float) -> tuple[np.ndarray, dict[str, int]]:
    """Return (drop_mask (N,), per_slug_dropped {slug: n}). Missing slugs
    are skipped with a WARN. Length mismatch is a hard error.
    """
    drop_mask = np.zeros(n_rows, dtype=bool)
    per_slug = {}
    for slug in slugs:
        arr = residue_scores.get(slug)
        if arr is None:
            _log(f"  [WARN] residue_scores missing slug '{slug}' -- skipped")
            per_slug[slug] = 0
            continue
        if arr.shape[0] != n_rows:
            raise ValueError(
                f"residue score for '{slug}' has {arr.shape[0]} entries "
                f"but current PLY has {n_rows} rows -- Agent A must "
                f"re-project against the current file")
        # NaN safe: NaN -> not flagged
        arr = np.nan_to_num(arr, nan=0.0)
        this_hit = arr > threshold
        per_slug[slug] = int(this_hit.sum())
        drop_mask |= this_hit
    return drop_mask, per_slug


# ---------------------------------------------------------------------------
# Patch-protection helpers (Fix 1 -- 2026-07-20)
# ---------------------------------------------------------------------------

def build_patch_protection_set(patch_dir: Path,
                                prompts: list,
                                dp_R: np.ndarray,
                                dp_t: np.ndarray,
                                dp_scale: float,
                                colmap_to_metric: float) -> np.ndarray:
    """Load all `desk_patch_<slug>.ply` centres from `patch_dir`, convert
    to METRIC metres, and return as a single (M, 3) float32 array.

    These are the intentional desk-patch fill Gaussians produced by
    `_seed_desk_patch.py`. The iterative refinement loop must never
    drop them, even when they happen to project into the SAM3 mask of
    a removed object. Callers KDTree this array (in metric space) and
    query at 1 mm tolerance.

    Returns an EMPTY (0, 3) float32 array if no patch files are found;
    callers should treat that as "no protection needed" rather than an
    error.
    """
    from plyfile import PlyData
    patch_dir = Path(patch_dir)
    all_pts_metric = []
    for slug in prompts:
        p = patch_dir / f"desk_patch_{slug}.ply"
        if not p.is_file():
            _log(f"  [patch-protect] {p.name}: MISSING -- skipped")
            continue
        pd = PlyData.read(str(p))
        v = pd["vertex"]
        xyz_splat = np.stack([np.asarray(v["x"], dtype=np.float64),
                              np.asarray(v["y"], dtype=np.float64),
                              np.asarray(v["z"], dtype=np.float64)], axis=-1)
        xyz_m = _splat_to_metric(xyz_splat, dp_R, dp_t, dp_scale,
                                 colmap_to_metric).astype(np.float32)
        all_pts_metric.append(xyz_m)
        _log(f"  [patch-protect] {p.name}: {len(xyz_m)} Gaussians loaded")
        del v, pd
    gc.collect()
    if not all_pts_metric:
        return np.zeros((0, 3), dtype=np.float32)
    out = np.concatenate(all_pts_metric, axis=0)
    _log(f"  [patch-protect] total protected positions: {len(out)}")
    return out


def _build_patch_tree(patch_positions):
    """Build a cKDTree over (M, 3) metric patch positions, or return
    None if patch_positions is None/empty. Import scipy lazily.
    """
    if patch_positions is None or len(patch_positions) == 0:
        return None
    from scipy.spatial import cKDTree
    return cKDTree(np.ascontiguousarray(patch_positions, dtype=np.float64))


def _patch_protection_mask(pos_m: np.ndarray,
                            patch_tree,
                            tol_m: float = 0.001) -> np.ndarray:
    """(N,) bool: True for each metric position within `tol_m` of ANY
    patch position. Returns an all-False mask if patch_tree is None.
    """
    n = int(pos_m.shape[0])
    if patch_tree is None:
        return np.zeros(n, dtype=bool)
    # query returns dist=inf when nothing is within upper bound
    dist, _ = patch_tree.query(np.ascontiguousarray(pos_m, dtype=np.float64),
                                k=1, distance_upper_bound=float(tol_m))
    return np.isfinite(dist)


# ---------------------------------------------------------------------------
# Colour-reference helpers (Fix 2 -- 2026-07-20)
# ---------------------------------------------------------------------------

def _mesh_median_rgb01(mesh_path: Path):
    """Load a mesh and return its median vertex colour in [0, 1] RGB, or
    None if the mesh has no usable per-vertex colours (and its texture
    visual can't be converted). Handles both `.obj` (with MTL+PNG) and
    textured `.ply` variants via trimesh's TextureVisuals -> ColorVisuals
    conversion.
    """
    import trimesh
    mesh_path = Path(mesh_path)
    m = trimesh.load(str(mesh_path), force="mesh", process=False)
    vc = None
    visual = getattr(m, "visual", None)
    # Preferred path: TextureVisuals.to_color() gives per-vertex RGBA
    if visual is not None and hasattr(visual, "to_color"):
        try:
            cv = visual.to_color()
            cand = np.asarray(getattr(cv, "vertex_colors", None))
            if cand is not None and cand.ndim == 2 and cand.shape[0] == len(m.vertices):
                vc = cand
        except Exception as _e:
            vc = None
    # Fallback: existing vertex_colors on visual
    if vc is None and visual is not None and hasattr(visual, "vertex_colors"):
        cand = np.asarray(visual.vertex_colors)
        if cand.ndim == 2 and cand.shape[0] == len(m.vertices):
            vc = cand
    if vc is None or len(vc) == 0:
        return None
    rgb = np.asarray(vc[:, :3], dtype=np.float64) / 255.0
    # Median is more robust than mean against occasional texture-atlas
    # black borders / seam pixels.
    med = np.median(rgb, axis=0)
    return tuple(float(x) for x in np.clip(med, 0.0, 1.0))


def _distance_to_aabb(pts: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Euclidean distance from each 3-vector in pts to the (lo, hi) AABB.
    Zero for points inside/on the box.
    """
    d = np.maximum(np.maximum(lo[None, :] - pts, 0.0),
                   pts - hi[None, :])
    return np.linalg.norm(d, axis=1)


def compute_color_references(prompts: list,
                              object_mesh_dir: Path,
                              scene_mesh_path: Path,
                              object_aabbs: dict,
                              *,
                              desk_planes: dict | None = None,
                              ring_min_m: float = 0.15,
                              ring_max_m: float = 0.30,
                              plane_tol_m: float = 0.03) -> dict:
    """Return {slug: {"obj_rgb": (r,g,b), "desk_rgb": (r,g,b)}} in [0, 1].

    obj_rgb : median vertex colour of `<slug>/<slug>_extracted.obj`
              (the OBJ carries its own MTL+PNG; the PLY sibling in
              this repo references a *shared* atlas so it is a poor
              per-object colour proxy).
    desk_rgb : median vertex colour of scene-mesh verts that are
               (a) between `ring_min_m` and `ring_max_m` OUTSIDE the
                   (padded) object AABB, AND
               (b) within `plane_tol_m` of the fitted desk plane
                   for that slug (if a plane is available in
                   `desk_planes`; otherwise only (a) is applied).

    Missing / colourless slugs are skipped and reported. This function
    is called ONCE at the start of the iterative loop -- it is O(scene
    mesh vert count) but no per-view projections.
    """
    import trimesh
    object_mesh_dir = Path(object_mesh_dir)
    scene_mesh_path = Path(scene_mesh_path) if scene_mesh_path else None

    # --- object colours -----------------------------------------------------
    obj_colors = {}
    for slug in prompts:
        obj_obj = object_mesh_dir / slug / f"{slug}_extracted.obj"
        obj_ply = object_mesh_dir / slug / f"{slug}_extracted.ply"
        picked = obj_obj if obj_obj.is_file() else obj_ply
        if not picked.is_file():
            _log(f"  [color-ref] {slug}: NO extracted mesh at "
                 f"{obj_obj} or {obj_ply} -- skipped")
            continue
        rgb = _mesh_median_rgb01(picked)
        if rgb is None:
            _log(f"  [color-ref] {slug}: {picked.name} has no usable "
                 f"vertex colours -- skipped")
            continue
        obj_colors[slug] = rgb
        _log(f"  [color-ref] {slug}: obj_rgb={tuple(round(c, 3) for c in rgb)}  "
             f"(from {picked.name})")

    # --- desk colours (need scene mesh) ------------------------------------
    out = {}
    if scene_mesh_path is None or not scene_mesh_path.is_file():
        _log(f"  [color-ref] scene_mesh_path missing "
             f"({scene_mesh_path}); returning obj-only refs (no colour filter)")
        for slug, rgb in obj_colors.items():
            out[slug] = {"obj_rgb": rgb, "desk_rgb": None}
        return out

    scene = trimesh.load(str(scene_mesh_path), force="mesh", process=False)
    scene_verts = np.asarray(scene.vertices, dtype=np.float64)
    scene_vc = None
    visual = getattr(scene, "visual", None)
    if visual is not None and hasattr(visual, "to_color"):
        try:
            cv = visual.to_color()
            cand = np.asarray(getattr(cv, "vertex_colors", None))
            if cand is not None and cand.ndim == 2 and cand.shape[0] == len(scene_verts):
                scene_vc = cand
        except Exception:
            scene_vc = None
    if scene_vc is None and visual is not None and hasattr(visual, "vertex_colors"):
        cand = np.asarray(visual.vertex_colors)
        if cand.ndim == 2 and cand.shape[0] == len(scene_verts):
            scene_vc = cand
    if scene_vc is None:
        _log(f"  [color-ref] {scene_mesh_path.name}: no per-vertex colours "
             f"(and TextureVisuals.to_color() failed) -- desk_rgb=None for "
             f"all slugs (colour filter will be skipped)")
        for slug, rgb in obj_colors.items():
            out[slug] = {"obj_rgb": rgb, "desk_rgb": None}
        return out
    scene_rgb = np.asarray(scene_vc[:, :3], dtype=np.float64) / 255.0

    for slug in prompts:
        if slug not in obj_colors:
            continue
        aabb = object_aabbs.get(slug)
        if aabb is None:
            _log(f"  [color-ref] {slug}: no AABB -- desk_rgb=None")
            out[slug] = {"obj_rgb": obj_colors[slug], "desk_rgb": None}
            continue
        lo = np.asarray(aabb["lo"], dtype=np.float64).reshape(3)
        hi = np.asarray(aabb["hi"], dtype=np.float64).reshape(3)
        dist = _distance_to_aabb(scene_verts, lo, hi)
        ring = (dist >= ring_min_m) & (dist <= ring_max_m)
        if desk_planes is not None and slug in desk_planes:
            plane = desk_planes[slug]
            n = np.asarray(plane["n"], dtype=np.float64).reshape(3)
            d = float(plane["d"])
            plane_sd = scene_verts @ n + d
            on_plane = np.abs(plane_sd) <= plane_tol_m
            keep = ring & on_plane
        else:
            keep = ring
        n_keep = int(keep.sum())
        if n_keep < 32:
            _log(f"  [color-ref] {slug}: only {n_keep} desk-ring verts "
                 f"({ring_min_m*100:.0f}-{ring_max_m*100:.0f} cm from AABB"
                 f"{', on plane' if desk_planes and slug in desk_planes else ''})"
                 f" -- desk_rgb=None (too sparse for a stable median)")
            out[slug] = {"obj_rgb": obj_colors[slug], "desk_rgb": None}
            continue
        med = np.median(scene_rgb[keep], axis=0)
        desk_rgb = tuple(float(c) for c in np.clip(med, 0.0, 1.0))
        out[slug] = {"obj_rgb": obj_colors[slug], "desk_rgb": desk_rgb}
        _log(f"  [color-ref] {slug}: desk_rgb={tuple(round(c, 3) for c in desk_rgb)}  "
             f"(from {n_keep} scene-mesh verts)")
    return out


def _gaussian_rgb01(rows: np.ndarray) -> np.ndarray | None:
    """Convert splat rows' f_dc_0/1/2 SH DC coefficients to linear
    [0, 1] RGB. Returns None if the columns are missing.
    """
    for f in ("f_dc_0", "f_dc_1", "f_dc_2"):
        if f not in rows.dtype.names:
            return None
    f_dc = np.stack([np.asarray(rows["f_dc_0"], dtype=np.float32),
                     np.asarray(rows["f_dc_1"], dtype=np.float32),
                     np.asarray(rows["f_dc_2"], dtype=np.float32)], axis=-1)
    rgb = 0.5 + f_dc * SH_C0
    return np.clip(rgb, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 1) Drop residue Gaussians (splat)
# ---------------------------------------------------------------------------

def drop_residue_gaussians(
    splat_path: Path,
    residue_scores: dict,
    *,
    threshold: float = 0.4,
    out_path: Path,
    snapshot_dir: Optional[Path] = None,
    patch_positions: Optional[np.ndarray] = None,
    color_refs: Optional[dict] = None,
    desk_planes: Optional[dict] = None,
    color_margin: float = 0.85,
    above_plane_drop_m: float = 0.03,
    dp_R: Optional[np.ndarray] = None,
    dp_t: Optional[np.ndarray] = None,
    dp_scale: Optional[float] = None,
    colmap_to_metric: Optional[float] = None,
) -> dict:
    """Drop Gaussians whose residue score exceeds `threshold` AND satisfy
    either the Y-above-plane test or the tightened colour test for their
    slug.

    residue_scores : {slug -> (N,) float in [0, 1]}. Row count MUST match
                     the current PLY row count.
    patch_positions: optional (M, 3) METRIC array of desk-patch positions
                     to protect. Any Gaussian within 1 mm of a patch pos
                     is never dropped. Requires the dp_R/dp_t/dp_scale/
                     colmap_to_metric transform quartet to convert splat
                     rows to metric metres.
    color_refs     : optional {slug: {"obj_rgb":..., "desk_rgb":...}}
                     from compute_color_references(). If provided AND
                     the slug has a non-None desk_rgb, the colour test
                     for that slug is
                         desk_closer := d_obj < color_margin * d_desk
                     (Euclidean in linear [0,1] RGB). At color_margin=1
                     this matches the v1 "closer to obj than desk" test;
                     the default 0.85 requires the Gaussian to be
                     CLEARLY closer to obj, which rejects the ambiguous
                     middle-ground colours v2 kept.
    desk_planes    : optional {slug: {"n": (3,), "d": float}} metric
                     world-space planes. If provided for a slug, any
                     residue-flagged Gaussian more than
                     `above_plane_drop_m` above the plane (signed
                     distance, up-normal convention) is DROPPED
                     regardless of colour.
    color_margin   : see color_refs above. Ignored when color_refs is
                     None for a slug.
    above_plane_drop_m : signed distance above the desk plane at which
                     "above the desk" starts. Default 3 cm.

    Combined drop rule per slug (when EITHER plane or colour info is
    available for that slug):
        slug_drop = residue_hit & (above_plane | desk_closer)
    When NEITHER is available we fall back to the v1 behaviour
    (drop every residue_hit for that slug). Patch protection is then
    applied to the OR-union across slugs -- Gaussians within 1 mm of a
    patch position are never dropped, even if they satisfy the drop
    rule for one or more slugs.
    """
    splat_path = Path(splat_path)
    out_path = Path(out_path)
    _log(f"[drop_residue_gaussians] {splat_path.name} threshold={threshold} "
         f"color_margin={color_margin} above_plane_drop_m={above_plane_drop_m:.3f}m")

    rows, _, n_in = _load_splat_rows(splat_path)
    slugs = list(residue_scores.keys())

    # Precompute Gaussian RGB from SH DC only if colour filter is active.
    gauss_rgb = None
    if color_refs:
        gauss_rgb = _gaussian_rgb01(rows)
        if gauss_rgb is None:
            _log(f"  [WARN] {splat_path.name} has no f_dc_0/1/2 columns; "
                 f"colour filter DISABLED for this call")

    # Metric positions are needed by both the Y-above-plane test and by
    # patch protection. Compute once (best-effort) so we don't pay the
    # cost twice per iter.
    need_pos = (
        (desk_planes is not None and len(desk_planes) > 0)
        or (patch_positions is not None and len(patch_positions))
    )
    pos_m = None
    if need_pos:
        if (dp_R is None or dp_t is None or dp_scale is None
                or colmap_to_metric is None):
            _log("  [WARN] desk_planes or patch_positions given but "
                 "transform quartet (dp_R/dp_t/dp_scale/colmap_to_metric) "
                 "missing; SKIPPING plane / patch checks this pass")
        else:
            pos_m = _splat_positions_metric(
                rows, dp_R=dp_R, dp_t=dp_t, dp_scale=dp_scale,
                colmap_to_metric=colmap_to_metric)

    drop = np.zeros(n_in, dtype=bool)
    per_slug_drop_masks: dict = {}
    per_slug_flagged: dict = {}
    per_slug_above_plane: dict = {}
    per_slug_color: dict = {}
    per_slug_dropped: dict = {}
    per_slug_patch_saved: dict = {}

    for slug in slugs:
        arr = residue_scores.get(slug)
        if arr is None:
            _log(f"  [WARN] residue_scores missing slug '{slug}' -- skipped")
            per_slug_drop_masks[slug] = np.zeros(n_in, dtype=bool)
            per_slug_flagged[slug] = 0
            per_slug_above_plane[slug] = 0
            per_slug_color[slug] = 0
            per_slug_dropped[slug] = 0
            per_slug_patch_saved[slug] = 0
            continue
        if arr.shape[0] != n_in:
            raise ValueError(
                f"residue score for '{slug}' has {arr.shape[0]} entries "
                f"but current PLY has {n_in} rows -- Agent A must "
                f"re-project against the current file")
        arr = np.nan_to_num(arr, nan=0.0)
        residue_hit = arr > threshold
        per_slug_flagged[slug] = int(residue_hit.sum())

        # ---- Y-above-plane test ---------------------------------------
        above_plane = np.zeros(n_in, dtype=bool)
        plane_available = (
            pos_m is not None
            and desk_planes is not None
            and slug in desk_planes
        )
        if plane_available:
            plane = desk_planes[slug]
            n_vec = np.asarray(plane["n"], dtype=np.float64).reshape(3)
            d_val = float(plane["d"])
            if n_vec[1] <= 0:
                raise ValueError(
                    f"desk plane for '{slug}' has n_y={n_vec[1]:.4f} <= 0 "
                    f"(convention requires up-normal). Refusing to run "
                    f"lest 'above desk' silently invert.")
            signed_dist = pos_m @ n_vec + d_val
            above_plane = signed_dist > float(above_plane_drop_m)

        # ---- Tightened colour test ------------------------------------
        desk_closer = np.zeros(n_in, dtype=bool)
        color_filter_active = (
            color_refs is not None and gauss_rgb is not None
            and slug in color_refs
            and color_refs[slug].get("desk_rgb") is not None
            and color_refs[slug].get("obj_rgb") is not None
        )
        if color_filter_active:
            obj_rgb = np.asarray(color_refs[slug]["obj_rgb"], dtype=np.float32)
            desk_rgb = np.asarray(color_refs[slug]["desk_rgb"], dtype=np.float32)
            d_obj = np.linalg.norm(gauss_rgb - obj_rgb[None, :], axis=1)
            d_desk = np.linalg.norm(gauss_rgb - desk_rgb[None, :], axis=1)
            desk_closer = d_obj < (float(color_margin) * d_desk)

        # ---- Combine --------------------------------------------------
        if plane_available or color_filter_active:
            slug_drop = residue_hit & (above_plane | desk_closer)
        else:
            # No new filter available -- fall back to v1 behaviour so we
            # don't silently become a no-op for slugs missing metadata.
            _log(f"  [WARN] {slug}: neither desk plane nor colour ref "
                 f"available -- falling back to drop-all-residue-hits")
            slug_drop = residue_hit.copy()

        per_slug_drop_masks[slug] = slug_drop
        per_slug_above_plane[slug] = int((residue_hit & above_plane).sum())
        per_slug_color[slug] = int((residue_hit & desk_closer).sum())
        drop |= slug_drop

    # ---- Patch protection ------------------------------------------------
    protect_mask = None
    if patch_positions is not None and len(patch_positions):
        if pos_m is None:
            _log("  [WARN] patch_positions given but pos_m unavailable; "
                 "SKIPPING patch protection this pass")
        else:
            tree = _build_patch_tree(patch_positions)
            protect_mask = _patch_protection_mask(pos_m, tree, tol_m=0.001)

    n_patch_saved = 0
    if protect_mask is not None:
        n_patch_saved = int((drop & protect_mask).sum())
        drop &= ~protect_mask
        for slug, mask_slug in per_slug_drop_masks.items():
            per_slug_patch_saved[slug] = int((mask_slug & protect_mask).sum())
            per_slug_dropped[slug] = int((mask_slug & ~protect_mask).sum())
    else:
        for slug, mask_slug in per_slug_drop_masks.items():
            per_slug_patch_saved[slug] = 0
            per_slug_dropped[slug] = int(mask_slug.sum())

    keep = ~drop
    n_out = int(keep.sum())
    n_dropped = int(drop.sum())

    _snapshot(splat_path, snapshot_dir, "drop_residue_gaussians")
    _atomic_write_ply(rows[keep], out_path)

    _log(f"  n_in={n_in}  n_dropped={n_dropped}  n_out={n_out}  "
         f"n_patch_saved={n_patch_saved}")
    for slug in slugs:
        _log(f"    {slug}: flagged={per_slug_flagged.get(slug, 0)}  "
             f"above_plane={per_slug_above_plane.get(slug, 0)}  "
             f"color={per_slug_color.get(slug, 0)}  "
             f"dropped={per_slug_dropped.get(slug, 0)}  "
             f"patch_saved={per_slug_patch_saved.get(slug, 0)}")
    return {
        "n_in": n_in, "n_out": n_out, "n_dropped": n_dropped,
        "n_patch_saved": n_patch_saved,
        "per_slug_flagged": per_slug_flagged,
        "per_slug_above_plane": per_slug_above_plane,
        "per_slug_color": per_slug_color,
        "per_slug_dropped": per_slug_dropped,
        "per_slug_patch_saved": per_slug_patch_saved,
    }


# ---------------------------------------------------------------------------
# 2) Drop sub-plane Gaussians (splat, per-slug)
# ---------------------------------------------------------------------------

def drop_subplane_gaussians(
    splat_path: Path,
    desk_planes: dict,
    object_aabbs: dict,
    *,
    dp_R: np.ndarray,
    dp_t: np.ndarray,
    dp_scale: float,
    colmap_to_metric: float,
    y_offset_m: float = 0.05,
    out_path: Path,
    snapshot_dir: Optional[Path] = None,
    patch_positions: Optional[np.ndarray] = None,
) -> dict:
    """Drop Gaussians whose METRIC centre lies both:
        (a) inside a slug's padded AABB, AND
        (b) more than `y_offset_m` BELOW that slug's fitted desk plane
            (n . p + d  <  -y_offset_m , plane convention: n_y > 0).

    patch_positions: optional (M, 3) METRIC array. Any Gaussian whose
    metric centre is within 1 mm of a patch position is never dropped,
    even if it independently meets criteria (a)+(b).
    """
    splat_path = Path(splat_path)
    out_path = Path(out_path)
    _log(f"[drop_subplane_gaussians] {splat_path.name} y_offset={y_offset_m*100:.1f}cm")

    rows, _, n_in = _load_splat_rows(splat_path)
    pos_m = _splat_positions_metric(
        rows, dp_R=dp_R, dp_t=dp_t, dp_scale=dp_scale,
        colmap_to_metric=colmap_to_metric)

    drop = np.zeros(n_in, dtype=bool)
    per_slug = {}
    for slug in desk_planes.keys():
        plane = desk_planes[slug]
        aabb = object_aabbs.get(slug)
        if aabb is None:
            _log(f"  [WARN] no AABB for '{slug}' -- skipped")
            per_slug[slug] = 0
            continue
        n = np.asarray(plane["n"], dtype=np.float64).reshape(3)
        d = float(plane["d"])
        if n[1] <= 0:
            raise ValueError(
                f"desk plane for '{slug}' has n_y={n[1]:.4f} <= 0 "
                f"(convention requires up-normal). Refusing to run "
                f"lest 'below desk' silently invert.")
        lo = np.asarray(aabb["lo"], dtype=np.float64).reshape(3)
        hi = np.asarray(aabb["hi"], dtype=np.float64).reshape(3)

        signed = pos_m @ n + d
        in_aabb = ((pos_m >= lo).all(axis=1) & (pos_m <= hi).all(axis=1))
        this_drop = in_aabb & (signed < -y_offset_m)
        per_slug[slug] = int(this_drop.sum())
        drop |= this_drop

    # Patch protection ------------------------------------------------------
    n_patch_saved = 0
    if patch_positions is not None and len(patch_positions):
        tree = _build_patch_tree(patch_positions)
        protect = _patch_protection_mask(pos_m, tree, tol_m=0.001)
        n_patch_saved = int((drop & protect).sum())
        drop &= ~protect

    keep = ~drop
    n_out = int(keep.sum())
    n_dropped = int(drop.sum())

    _snapshot(splat_path, snapshot_dir, "drop_subplane_gaussians")
    _atomic_write_ply(rows[keep], out_path)

    _log(f"  n_in={n_in}  n_dropped={n_dropped}  n_out={n_out}  "
         f"n_patch_saved={n_patch_saved}")
    for s, n in per_slug.items():
        _log(f"    {s}: {n} sub-plane")
    return {
        "n_in": n_in, "n_out": n_out, "n_dropped": n_dropped,
        "n_patch_saved": n_patch_saved,
        "per_slug_dropped": per_slug,
    }


# ---------------------------------------------------------------------------
# 3) Drop needle (anisotropic long-axis) Gaussians in per-object regions
# ---------------------------------------------------------------------------

def drop_needle_gaussians_in_region(
    splat_path: Path,
    object_aabbs: dict,
    *,
    dp_R: np.ndarray,
    dp_t: np.ndarray,
    dp_scale: float,
    colmap_to_metric: float,
    splat_to_metric: float,
    max_axis_mm: float = 30.0,
    aniso_ratio: float = 5.0,
    out_path: Path,
    snapshot_dir: Optional[Path] = None,
    patch_positions: Optional[np.ndarray] = None,
) -> dict:
    """Drop Gaussians whose METRIC centre lies inside ANY slug's AABB
    AND that are simultaneously long (sigma_max_mm > max_axis_mm) AND
    anisotropic (sigma_max / sigma_min >= aniso_ratio).

    patch_positions: optional (M, 3) METRIC array. Any Gaussian within
    1 mm of a patch position is never dropped.
    """
    splat_path = Path(splat_path)
    out_path = Path(out_path)
    _log(f"[drop_needle_gaussians_in_region] {splat_path.name} "
         f"max_axis_mm={max_axis_mm} aniso={aniso_ratio}")

    rows, dtype, n_in = _load_splat_rows(splat_path)
    scale_fields = sorted(n for n in dtype.names if n.startswith("scale_"))
    if len(scale_fields) < 3:
        raise ValueError(
            f"splat {splat_path} has {len(scale_fields)} scale_* fields; "
            f"need at least 3")
    scale_log = np.stack(
        [np.asarray(rows[f], dtype=np.float64) for f in scale_fields[:3]],
        axis=-1)
    sigma_splat = np.exp(scale_log)                                    # (N, 3)
    sigma_mm = sigma_splat * splat_to_metric * 1000.0                  # (N, 3)
    sigma_max_mm = sigma_mm.max(axis=1)
    sigma_min_mm = np.maximum(sigma_mm.min(axis=1), 1e-9)
    ratio = sigma_max_mm / sigma_min_mm

    is_needle = (sigma_max_mm > max_axis_mm) & (ratio >= aniso_ratio)

    pos_m = _splat_positions_metric(
        rows, dp_R=dp_R, dp_t=dp_t, dp_scale=dp_scale,
        colmap_to_metric=colmap_to_metric)

    in_any_aabb = np.zeros(n_in, dtype=bool)
    per_slug = {}
    for slug, aabb in object_aabbs.items():
        lo = np.asarray(aabb["lo"], dtype=np.float64).reshape(3)
        hi = np.asarray(aabb["hi"], dtype=np.float64).reshape(3)
        this_in = ((pos_m >= lo).all(axis=1) & (pos_m <= hi).all(axis=1))
        drop_this = this_in & is_needle
        per_slug[slug] = int(drop_this.sum())
        in_any_aabb |= this_in

    drop = in_any_aabb & is_needle

    # Patch protection ------------------------------------------------------
    n_patch_saved = 0
    if patch_positions is not None and len(patch_positions):
        tree = _build_patch_tree(patch_positions)
        protect = _patch_protection_mask(pos_m, tree, tol_m=0.001)
        n_patch_saved = int((drop & protect).sum())
        drop &= ~protect

    keep = ~drop
    n_out = int(keep.sum())
    n_dropped = int(drop.sum())

    _snapshot(splat_path, snapshot_dir, "drop_needle_gaussians_in_region")
    _atomic_write_ply(rows[keep], out_path)

    _log(f"  n_in={n_in}  n_dropped={n_dropped}  n_out={n_out}  "
         f"n_patch_saved={n_patch_saved}")
    for s, n in per_slug.items():
        _log(f"    {s}: {n} needle-in-AABB")
    return {
        "n_in": n_in, "n_out": n_out, "n_dropped": n_dropped,
        "n_patch_saved": n_patch_saved,
        "per_slug_dropped": per_slug,
    }


# ---------------------------------------------------------------------------
# 4) Drop residue mesh vertices (any-vertex-in-face test)
# ---------------------------------------------------------------------------

def drop_residue_mesh_verts(
    mesh_path: Path,
    residue_scores: dict,
    *,
    threshold: float = 0.4,
    out_path: Path,
    snapshot_dir: Optional[Path] = None,
    patch_positions: Optional[np.ndarray] = None,
) -> dict:
    """Drop mesh faces where ANY of the 3 vertices scores above the
    residue threshold for ANY slug. Unreferenced vertices are compacted.
    Vertex colours are preserved via the remap trick.

    patch_positions: optional (M, 3) METRIC array. Any mesh vertex
    within 1 mm of a patch position is NEVER flagged as residue,
    even if it independently scores above threshold. Note that patch
    positions were seeded into the SPLAT, not the mesh -- so in
    practice this rarely triggers, but it costs almost nothing and
    keeps the guarantee symmetric with the splat side.
    """
    import trimesh
    mesh_path = Path(mesh_path)
    out_path = Path(out_path)
    _log(f"[drop_residue_mesh_verts] {mesh_path.name} threshold={threshold}")

    m = trimesh.load(str(mesh_path), force="mesh", process=False)
    verts = np.asarray(m.vertices, dtype=np.float64)
    faces = np.asarray(m.faces, dtype=np.int64)
    n_verts_in = len(verts)
    n_faces_in = len(faces)

    drop_vert, per_slug = _bulk_or_scores(
        residue_scores, n_verts_in, list(residue_scores.keys()), threshold)

    # Patch protection ------------------------------------------------------
    n_patch_saved_verts = 0
    if patch_positions is not None and len(patch_positions):
        tree = _build_patch_tree(patch_positions)
        protect = _patch_protection_mask(
            verts.astype(np.float32), tree, tol_m=0.001)
        n_patch_saved_verts = int((drop_vert & protect).sum())
        drop_vert &= ~protect

    face_drop = drop_vert[faces].any(axis=1)
    face_keep = ~face_drop
    kept_faces = faces[face_keep]
    per_slug_faces = {}
    for slug, arr in residue_scores.items():
        if arr is None or arr.shape[0] != n_verts_in:
            per_slug_faces[slug] = 0
            continue
        this_vert = np.nan_to_num(arr, nan=0.0) > threshold
        this_face = this_vert[faces].any(axis=1)
        per_slug_faces[slug] = int(this_face.sum())

    used = np.unique(kept_faces.flatten()) if len(kept_faces) else np.zeros(0, dtype=np.int64)
    remap = -np.ones(n_verts_in, dtype=np.int64)
    remap[used] = np.arange(len(used))
    new_verts = verts[used]
    new_faces = remap[kept_faces] if len(kept_faces) else np.zeros((0, 3), dtype=np.int64)

    new_colors = None
    visual = getattr(m, "visual", None)
    if visual is not None and hasattr(visual, "vertex_colors"):
        vc = np.asarray(visual.vertex_colors)
        if vc.shape[0] == n_verts_in:
            new_colors = vc[used]

    if new_colors is not None:
        trimmed = trimesh.Trimesh(vertices=new_verts, faces=new_faces,
                                  vertex_colors=new_colors, process=False)
    else:
        trimmed = trimesh.Trimesh(vertices=new_verts, faces=new_faces,
                                  process=False)

    _snapshot(mesh_path, snapshot_dir, "drop_residue_mesh_verts")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    # trimesh infers exporter from suffix; the ".tmp" tail hides the real
    # ".ply", so pass the source suffix explicitly.
    tmesh_file_type = out_path.suffix.lstrip(".").lower() or "ply"
    trimmed.export(str(tmp), file_type=tmesh_file_type)
    gc.collect()
    try:
        os.replace(tmp, out_path)
    except PermissionError:
        gc.collect()
        try:
            os.replace(tmp, out_path)
        except PermissionError:
            if out_path.exists():
                try:
                    out_path.unlink()
                except PermissionError:
                    gc.collect()
                    out_path.unlink()
            os.rename(tmp, out_path)

    n_verts_out = len(new_verts)
    n_faces_out = len(new_faces)
    _log(f"  verts {n_verts_in}->{n_verts_out}  "
         f"faces {n_faces_in}->{n_faces_out}  "
         f"(dropped {n_faces_in - n_faces_out} faces)  "
         f"n_patch_saved_verts={n_patch_saved_verts}")
    for s, n in per_slug_faces.items():
        _log(f"    {s}: {n} faces flagged (OR-union may exceed final drop)")

    return {
        "n_verts_in": n_verts_in,
        "n_verts_out": n_verts_out,
        "n_faces_in": n_faces_in,
        "n_faces_out": n_faces_out,
        "n_patch_saved_verts": n_patch_saved_verts,
        "per_slug_faces_dropped": per_slug_faces,
    }


# ---------------------------------------------------------------------------
# 5) Densify patch Gaussians (splat)
# ---------------------------------------------------------------------------

def _project_to_plane(x: np.ndarray, z: np.ndarray,
                      n: np.ndarray, d: float) -> np.ndarray:
    """Given XZ coordinates and desk plane (n . p + d = 0), return Y.
    Requires n_y != 0."""
    ny = float(n[1])
    if abs(ny) < 1e-9:
        raise ValueError("desk plane n_y ~ 0; cannot project XZ->Y")
    return -(float(n[0]) * x + float(n[2]) * z + d) / ny


def densify_patch_gaussians(
    splat_path: Path,
    patch_regions: dict,
    *,
    dp_R: np.ndarray,
    dp_t: np.ndarray,
    dp_scale: float,
    colmap_to_metric: float,
    splat_to_metric: float,
    min_density_per_cm2: float = 5.0,
    clone_radius_m: float = 0.02,
    max_added_per_cell: int = 8,
    max_added_total: int = 5000,
    opacity_init: float = 0.9,
    scale_init_mm: float = 4.0,
    out_path: Path,
    snapshot_dir: Optional[Path] = None,
) -> dict:
    """Clone nearest existing Gaussian into sparse desk-patch cells.

    For each slug's patch footprint (irregular XZ grid on the fitted
    desk plane), we bin the existing Gaussians whose METRIC XZ lies in
    the mask2d region into the same (nx, nz) grid. For every cell whose
    count / cell_area_cm2 < min_density_per_cm2 we:

        (a) find the nearest existing Gaussian in the same slug whose
            metric XZ is within `clone_radius_m`;
        (b) clone its SH + rotation, place centre at the cell's XZ
            projected to the desk plane, and overwrite opacity/scale
            with the caller's inits.

    Cells with no donor within `clone_radius_m` are skipped and
    reported. New rows are appended (row-order-append) and Agent C's
    next residue pass MUST re-project (row count grows).
    """
    from scipy.spatial import cKDTree
    splat_path = Path(splat_path)
    out_path = Path(out_path)
    _log(f"[densify_patch_gaussians] {splat_path.name}  "
         f"min_density={min_density_per_cm2}/cm^2  "
         f"max_added_total={max_added_total}")

    rows, dtype, n_in = _load_splat_rows(splat_path)
    pos_m = _splat_positions_metric(
        rows, dp_R=dp_R, dp_t=dp_t, dp_scale=dp_scale,
        colmap_to_metric=colmap_to_metric)

    # KDTree over ALL existing Gaussian XZ for cheap donor lookup
    tree_all = cKDTree(pos_m[:, [0, 2]])

    new_rows_list = []
    per_slug = {}
    total_added = 0

    for slug, reg in patch_regions.items():
        pts_m = np.asarray(reg["pts_m"], dtype=np.float64)     # (K, 3)
        bounds_xz = reg["bounds_xz"]                             # (x0,x1,z0,z1)
        grid_res = float(reg.get("grid_res", 0.005))
        plane = reg.get("desk_plane")
        if plane is None:
            _log(f"  [WARN] {slug}: no desk_plane -- skipped")
            per_slug[slug] = {
                "cells_sparse": 0,
                "cells_donor_missing": 0,
                "added": 0,
            }
            continue
        n_plane = np.asarray(plane["n"], dtype=np.float64).reshape(3)
        d_plane = float(plane["d"])
        if n_plane[1] <= 0:
            raise ValueError(
                f"desk plane for '{slug}' has n_y={n_plane[1]:.4f} <= 0")

        x0, x1, z0, z1 = [float(v) for v in bounds_xz]
        xs = np.arange(x0, x1 + grid_res * 0.5, grid_res)
        zs = np.arange(z0, z1 + grid_res * 0.5, grid_res)
        nx, nz = len(xs), len(zs)
        cell_area_cm2 = (grid_res * 100.0) ** 2

        # Which mask2d cells are IN the footprint?
        mask2d = np.asarray(reg["mask2d"], dtype=bool)
        if mask2d.shape != (nx, nz):
            _log(f"  [WARN] {slug}: mask2d shape {mask2d.shape} != "
                 f"({nx},{nz}) -- skipped")
            per_slug[slug] = {
                "cells_sparse": 0,
                "cells_donor_missing": 0,
                "added": 0,
            }
            continue

        # Existing Gaussians in the enclosing rectangle
        in_rect = ((pos_m[:, 0] >= x0) & (pos_m[:, 0] <= x1)
                   & (pos_m[:, 2] >= z0) & (pos_m[:, 2] <= z1))
        rect_idx = np.nonzero(in_rect)[0]
        cell_count = np.zeros((nx, nz), dtype=np.int32)
        if len(rect_idx):
            cx = ((pos_m[rect_idx, 0] - x0) / grid_res).astype(int)
            cz = ((pos_m[rect_idx, 2] - z0) / grid_res).astype(int)
            in_bounds = ((cx >= 0) & (cx < nx) & (cz >= 0) & (cz < nz))
            cx = cx[in_bounds]
            cz = cz[in_bounds]
            np.add.at(cell_count, (cx, cz), 1)

        threshold_count = max(1, int(np.ceil(
            min_density_per_cm2 * cell_area_cm2)))
        sparse = mask2d & (cell_count < threshold_count)
        n_sparse = int(sparse.sum())

        n_donor_missing = 0
        n_added_slug = 0
        target_per_cell = threshold_count  # what we aim to reach per sparse cell

        for ix, iz in zip(*np.nonzero(sparse)):
            if total_added >= max_added_total:
                break
            need = min(max_added_per_cell,
                       max(1, target_per_cell - int(cell_count[ix, iz])))
            xm = x0 + (ix + 0.5) * grid_res
            zm = z0 + (iz + 0.5) * grid_res
            ym = _project_to_plane(np.array([xm]), np.array([zm]),
                                    n_plane, d_plane)[0]
            # Find nearest existing donor in XZ within clone_radius_m
            dist, donor_idx = tree_all.query(
                np.array([xm, zm]), k=1,
                distance_upper_bound=float(clone_radius_m))
            if not np.isfinite(dist) or donor_idx >= n_in:
                n_donor_missing += 1
                continue
            for _ in range(need):
                if total_added >= max_added_total:
                    break
                r = np.zeros(1, dtype=dtype)
                # Clone donor SH + rotation + normals
                r[0] = rows[donor_idx]
                # Overwrite geometry
                # Forward transform metric -> splat:
                #   splat = dp_scale * (R @ (metric/colmap_to_metric) + t)
                metric_pt = np.array([[xm, ym, zm]], dtype=np.float64)
                colmap_pt = metric_pt / colmap_to_metric
                splat_pt = dp_scale * (colmap_pt @ dp_R.T + dp_t[None, :])
                r[0]["x"] = np.float32(splat_pt[0, 0])
                r[0]["y"] = np.float32(splat_pt[0, 1])
                r[0]["z"] = np.float32(splat_pt[0, 2])
                # Opacity (stored as pre-sigmoid logit)
                op_clip = float(np.clip(opacity_init, 1e-4, 1 - 1e-4))
                op_logit = float(np.log(op_clip / (1 - op_clip)))
                r[0]["opacity"] = np.float32(op_logit)
                # Scales stored as log(sigma) in SPLAT units
                sigma_splat = (scale_init_mm / 1000.0) / splat_to_metric
                s_log = float(np.log(max(sigma_splat, 1e-9)))
                for f in ("scale_0", "scale_1", "scale_2"):
                    if f in dtype.names:
                        r[0][f] = np.float32(s_log)
                new_rows_list.append(r[0])
                n_added_slug += 1
                total_added += 1

        per_slug[slug] = {
            "cells_sparse": n_sparse,
            "cells_donor_missing": n_donor_missing,
            "added": n_added_slug,
        }
        _log(f"  {slug}: sparse cells={n_sparse}  "
             f"donor_missing={n_donor_missing}  added={n_added_slug}")

    _snapshot(splat_path, snapshot_dir, "densify_patch_gaussians")

    if new_rows_list:
        new_rows = np.array(new_rows_list, dtype=dtype)
        out_rows = np.concatenate([rows, new_rows])
    else:
        out_rows = rows

    _atomic_write_ply(out_rows, out_path)
    n_out = int(len(out_rows))
    _log(f"  n_in={n_in}  n_added={total_added}  n_out={n_out}")

    return {
        "n_in": n_in,
        "n_added": total_added,
        "n_out": n_out,
        "per_slug": per_slug,
    }


# ---------------------------------------------------------------------------
# 6) Adjust per-slug crop distance (config-only, deferred)
# ---------------------------------------------------------------------------

def adjust_crop_dist_for_object(
    current_config: dict,
    slug: str,
    delta_m: float,
    *,
    min_m: float = 0.03,
    max_m: float = 0.15,
) -> dict:
    """Mutate current_config['per_slug_crop_dist_m'][slug] by delta_m,
    clamped to [min_m, max_m]. Config-only -- no PLY is touched. Agent C
    reads this dict on the NEXT full Stage-4 re-run.
    """
    if "per_slug_crop_dist_m" not in current_config:
        current_config["per_slug_crop_dist_m"] = {}
    d = current_config["per_slug_crop_dist_m"]
    prev = float(d.get(slug, 0.05))
    new = float(np.clip(prev + float(delta_m), min_m, max_m))
    d[slug] = new
    _log(f"[adjust_crop_dist] {slug}: {prev:.3f} m -> {new:.3f} m "
         f"(delta {delta_m:+.3f} m, clamped to [{min_m},{max_m}])")
    return current_config


# ---------------------------------------------------------------------------
# Object-AABB / desk-plane loader (convenience so Agent C stays thin)
# ---------------------------------------------------------------------------

def load_object_aabbs(object_mesh_dir: Path, slugs, pad_m: float = 0.07) -> dict:
    """Compute padded AABBs (metric metres) from each slug's extracted
    object mesh. `pad_m` should be at least crop_dist_m + a couple cm.
    """
    import trimesh
    object_mesh_dir = Path(object_mesh_dir)
    out = {}
    for slug in slugs:
        p = object_mesh_dir / slug / f"{slug}_extracted.ply"
        if not p.is_file():
            _log(f"  [WARN] load_object_aabbs: missing {p}")
            continue
        m = trimesh.load(str(p), force="mesh", process=False)
        v = np.asarray(m.vertices, dtype=np.float64)
        lo = v.min(axis=0) - pad_m
        hi = v.max(axis=0) + pad_m
        out[slug] = {"lo": lo, "hi": hi}
    return out


def load_desk_planes(unseen_dir: Path, slugs) -> dict:
    """Load fitted desk plane + y_desk fallback from unseen_core_<slug>.npz."""
    unseen_dir = Path(unseen_dir)
    out = {}
    for slug in slugs:
        p = unseen_dir / f"unseen_core_{slug}.npz"
        if not p.is_file():
            _log(f"  [WARN] load_desk_planes: missing {p}")
            continue
        data = np.load(p)
        if "desk_normal_metric" not in data.files or "desk_plane_d" not in data.files:
            _log(f"  [WARN] {p} lacks plane fields -- skipped")
            continue
        n = np.asarray(data["desk_normal_metric"], dtype=np.float64).reshape(3)
        d = float(np.asarray(data["desk_plane_d"], dtype=np.float64).flat[0])
        if n[1] < 0:
            # Auto-flip so the loader is friendly, but log loudly.
            _log(f"  [WARN] {slug}: desk normal points DOWN (n_y={n[1]:.3f}); "
                 f"auto-flipping to up-convention.")
            n = -n
            d = -d
        y_desk = None
        if "bounds_xyz" in data.files:
            b_arr = np.asarray(data["bounds_xyz"], dtype=np.float64)
            if b_arr.size >= 3:
                y_desk = float(b_arr[2])
        out[slug] = {"n": n, "d": d, "y_desk": y_desk}
    return out


def load_patch_regions(unseen_dir: Path, slugs) -> dict:
    """Load the (pts_m, mask2d, bounds_xz, grid_res, desk_plane) tuples
    for densify_patch_gaussians from the NPZs Stage 4 (re)wrote.
    """
    unseen_dir = Path(unseen_dir)
    out = {}
    for slug in slugs:
        p = unseen_dir / f"unseen_core_{slug}.npz"
        if not p.is_file():
            continue
        data = np.load(p)
        if "grid_points" not in data.files or "shape_xz" not in data.files \
                or "bounds_xyz" not in data.files:
            _log(f"  [WARN] {p} lacks footprint fields -- skipped")
            continue
        pts_m = np.asarray(data["grid_points"], dtype=np.float64)
        shape_xz = tuple(int(x) for x in np.asarray(data["shape_xz"]))
        nx, nz = shape_xz
        bounds_xyz = np.asarray(data["bounds_xyz"], dtype=np.float64)
        x_min, x_max, _y_desk, z_min, z_max = [float(v) for v in bounds_xyz]
        grid_res = 0.005
        if nx > 1:
            grid_res = (x_max - x_min) / (nx - 1)
        elif nz > 1:
            grid_res = (z_max - z_min) / (nz - 1)
        if "footprint_mask" in data.files:
            fp = np.asarray(data["footprint_mask"], dtype=bool).reshape(nx, nz)
        else:
            fp = np.ones((nx, nz), dtype=bool)
        plane = None
        if "desk_normal_metric" in data.files and "desk_plane_d" in data.files:
            n = np.asarray(data["desk_normal_metric"], dtype=np.float64).reshape(3)
            d = float(np.asarray(data["desk_plane_d"], dtype=np.float64).flat[0])
            if n[1] < 0:
                n = -n
                d = -d
            plane = {"n": n, "d": d}
        out[slug] = {
            "pts_m": pts_m,
            "mask2d": fp,
            "bounds_xz": (x_min, x_max, z_min, z_max),
            "grid_res": grid_res,
            "desk_plane": plane,
        }
    return out


if __name__ == "__main__":
    # Placeholder CLI. Agent C imports these functions directly; running
    # this module standalone just prints the available action list.
    print("Available surgical actions (import & call):")
    print("  drop_residue_gaussians")
    print("  drop_subplane_gaussians")
    print("  drop_needle_gaussians_in_region")
    print("  drop_residue_mesh_verts")
    print("  densify_patch_gaussians")
    print("  adjust_crop_dist_for_object")
    print("Loaders:")
    print("  load_object_aabbs, load_desk_planes, load_patch_regions")
