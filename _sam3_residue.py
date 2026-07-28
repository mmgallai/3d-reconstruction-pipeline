"""Agent A -- SAM3-based residue detector for the iterative refinement loop.

For every Gaussian centre (splat side) or mesh vertex (mesh side) we ask:
    "How often does this 3D point project INSIDE the cached SAM3 mask
     of each removed object, across all 184 posed views?"

If the fraction of visible views in which the point lands inside the
mask exceeds a threshold, the point is flagged as "residue" of that
object -- geometry that Stage-4 left behind after the object was
supposed to be removed. Agent B then decides whether to surgically
drop it.

The detector is stateless with respect to geometry -- it only needs:

  * per-view masks under `output/segmented_data4_v9a_fp_v2/masks/`
    named `<view_stem>__<slug>.png` (uint8, 0 / 255), pre-generated
    by scene_segmenter for the original (uncropped) scene photos.
  * `V32ViewSource` (the 184 registered Femto views).
  * an ndarray of METRIC world positions.

No SAM3 model is (re-)run.  Mask I/O + 8 px dilate are the dominant
cost; the loop is per-view so peak memory stays near ~55 MB for the
splat (~205 k Gaussians) and ~90 MB for the mesh (~881 k vertices).

Reused helpers (imported, not copied):
    _spatial_crop_splat._splat_to_metric
    _spatial_crop_splat._load_dataparser_transform
    scene_segmenter.views.V32ViewSource
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

from _spatial_crop_splat import (  # noqa: E402
    _load_dataparser_transform,
    _splat_to_metric,
)
from scene_segmenter.views import V32ViewSource  # noqa: E402


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResidueResult:
    """Per-point, per-slug residue score.

    residue_score[i, s] is the fraction of visible views in which point
    i projected inside the SAM3 mask for slug s. It is `NaN` when
    `visible_views[i] < min_views` (insufficient evidence).
    """
    residue_score: np.ndarray    # (N, S) float32, [0, 1] or NaN
    residue_hits: np.ndarray     # (N, S) int32 raw counts
    visible_views: np.ndarray    # (N,)   int32
    slugs: tuple                 # tuple[str, ...]

    def is_residue(self, threshold: float = 0.4) -> np.ndarray:
        """(N, S) bool. NaN scores treated as 0 (not flagged)."""
        return np.nan_to_num(self.residue_score, nan=0.0) > threshold

    def any_residue(self, threshold: float = 0.4) -> np.ndarray:
        """(N,) bool: True if flagged by ANY slug."""
        return self.is_residue(threshold).any(axis=1)

    def per_slug_score(self, slug: str) -> np.ndarray:
        """(N,) float32 -- convenience accessor by slug name."""
        idx = self.slugs.index(slug)
        return self.residue_score[:, idx]


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def _load_colmap_to_metric(bounds_path: Path) -> float:
    b = json.loads(Path(bounds_path).read_text())
    return 1.0 / float(b["scale_factor_da3_to_colmap"])


def _load_splat_positions_metric(scene_splat_path: Path,
                                 dp_json: Path,
                                 bounds_json: Path) -> np.ndarray:
    """Read splat PLY, inverse-transform x/y/z to metric metres.

    Windows note: plyfile mmap-reads the binary body by default and
    keeps the file handle live until the PlyData object is GC'd. If
    a repair step tries to os.replace this file later in the same
    Python process, it hits WinError 5. Explicitly drop references
    and force a GC so the handle releases before we return.
    """
    import gc as _gc
    from plyfile import PlyData
    pd = PlyData.read(str(scene_splat_path))
    v = pd["vertex"]
    xyz_splat = np.stack([np.asarray(v["x"], dtype=np.float64),
                          np.asarray(v["y"], dtype=np.float64),
                          np.asarray(v["z"], dtype=np.float64)], axis=-1)
    R, t, dp_scale = _load_dataparser_transform(Path(dp_json))
    c2m = _load_colmap_to_metric(Path(bounds_json))
    xyz_m = _splat_to_metric(xyz_splat, R, t, dp_scale, c2m)
    del v, pd
    _gc.collect()
    return xyz_m.astype(np.float32)


def _load_mesh_vertices_and_faces(scene_mesh_path: Path):
    import trimesh
    m = trimesh.load(str(scene_mesh_path), force="mesh", process=False)
    verts = np.asarray(m.vertices, dtype=np.float32)
    faces = np.asarray(m.faces, dtype=np.int64)
    return verts, faces


# ---------------------------------------------------------------------------
# Core: per-view projection loop
# ---------------------------------------------------------------------------

def compute_residue_scores(
    positions_metric: np.ndarray,
    view_source: V32ViewSource,
    sam3_cache_dir: Path,
    slugs: Sequence[str],
    *,
    min_views: int = 20,
    dilate_px: int = 8,
    z_near: float = 0.05,
    occlusion_mesh_path: Optional[Path] = None,
    depth_tol_m: float = 0.02,
    verbose: bool = True,
) -> ResidueResult:
    """Score every point in `positions_metric` (METRES) against every
    slug's cached SAM3 masks in `sam3_cache_dir`.

    See module docstring / design doc for full contract.
    """
    import cv2

    positions_metric = np.ascontiguousarray(positions_metric, dtype=np.float32)
    N = positions_metric.shape[0]
    S = len(slugs)
    sam3_cache_dir = Path(sam3_cache_dir)

    views = view_source.all_views()
    if not views:
        raise RuntimeError("view_source returned zero views")

    if verbose:
        print(f"[residue] positions={N}  slugs={list(slugs)}  "
              f"views={len(views)}  min_views={min_views}  "
              f"dilate_px={dilate_px}", flush=True)

    # Cheap sanity: first-view mask availability
    if verbose:
        v0 = views[0]
        found = sum(int((sam3_cache_dir / f"{v0.name}__{s}.png").is_file())
                    for s in slugs)
        print(f"[residue] first-view mask availability "
              f"({v0.name}): {found}/{len(slugs)}", flush=True)

    kernel = None
    if dilate_px > 0:
        kernel = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)

    # Optional occlusion scene (built once)
    occ_scene = None
    if occlusion_mesh_path is not None:
        import open3d as o3d
        import trimesh as _tm
        m = _tm.load(str(occlusion_mesh_path), force="mesh", process=False)
        tm = o3d.t.geometry.TriangleMesh()
        tm.vertex.positions = o3d.core.Tensor(
            np.asarray(m.vertices, dtype=np.float32))
        tm.triangle.indices = o3d.core.Tensor(
            np.asarray(m.faces, dtype=np.uint32))
        occ_scene = o3d.t.geometry.RaycastingScene()
        occ_scene.add_triangles(tm)
        if verbose:
            print(f"[residue] occlusion mesh loaded from "
                  f"{occlusion_mesh_path}", flush=True)

    visible_views = np.zeros(N, dtype=np.int32)
    residue_hits = np.zeros((N, S), dtype=np.int32)
    per_slug_missing = np.zeros(S, dtype=np.int32)

    t0 = time.time()
    for vi, view in enumerate(views):
        K = view.K
        w2c = view.w2c
        # OpenCV: pts_cam = R @ pts_world + t
        pts_cam = positions_metric @ w2c[:3, :3].T.astype(np.float32) \
            + w2c[:3, 3].astype(np.float32)
        zc = pts_cam[:, 2]
        in_front = zc > z_near
        zc_safe = np.where(in_front, zc, 1.0)
        u = K[0, 0] * pts_cam[:, 0] / zc_safe + K[0, 2]
        v = K[1, 1] * pts_cam[:, 1] / zc_safe + K[1, 2]
        in_frame = (in_front
                    & (u >= 0) & (u < view.width)
                    & (v >= 0) & (v < view.height))

        # Optional occlusion test (only for currently in-frame points)
        if occ_scene is not None and in_frame.any():
            import open3d as o3d
            cam_pos = view.c2w[:3, 3].astype(np.float32)
            dirs = positions_metric - cam_pos[None, :]
            dists = np.linalg.norm(dirs, axis=1)
            safe = np.maximum(dists, 1e-9)
            dirs_norm = dirs / safe[:, None]
            rays = np.concatenate(
                [np.broadcast_to(cam_pos, dirs_norm.shape),
                 dirs_norm], axis=1).astype(np.float32)
            hit_t = occ_scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
            unobstructed = hit_t >= dists - depth_tol_m
            in_frame &= unobstructed

        visible_views += in_frame.astype(np.int32)

        if not in_frame.any():
            continue

        idx = np.nonzero(in_frame)[0]
        u_int = np.clip(np.round(u[idx]).astype(np.int32), 0, view.width - 1)
        v_int = np.clip(np.round(v[idx]).astype(np.int32), 0, view.height - 1)

        for s_idx, slug in enumerate(slugs):
            p = sam3_cache_dir / f"{view.name}__{slug}.png"
            if not p.is_file():
                per_slug_missing[s_idx] += 1
                continue
            m_img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m_img is None:
                per_slug_missing[s_idx] += 1
                continue
            if kernel is not None:
                m_img = cv2.dilate(m_img, kernel, iterations=1)
            hit_local = m_img[v_int, u_int] > 128
            if hit_local.any():
                np.add.at(residue_hits[:, s_idx], idx[hit_local], 1)

        if verbose and ((vi + 1) % 40 == 0 or (vi + 1) == len(views)):
            dt = time.time() - t0
            print(f"[residue]   view {vi + 1}/{len(views)}  "
                  f"elapsed={dt:.1f}s", flush=True)

    # Post-process
    denom = np.maximum(visible_views, 1).astype(np.float32)[:, None]
    score = residue_hits.astype(np.float32) / denom
    insufficient = visible_views < min_views
    score[insufficient, :] = np.nan

    if verbose:
        dt = time.time() - t0
        print(f"[residue] finished in {dt:.1f}s", flush=True)
        print(f"[residue] insufficient-visibility rows "
              f"(<{min_views} views): {int(insufficient.sum())}/{N} "
              f"({100.0 * insufficient.mean():.1f}%)", flush=True)
        if visible_views.max() < 30:
            print("[residue] WARN visible_views.max() < 30 -- "
                  "likely coord-frame bug (splat not inverse-transformed?)",
                  flush=True)
        for s_idx, slug in enumerate(slugs):
            miss = int(per_slug_missing[s_idx])
            if miss > 0:
                print(f"[residue]   {slug}: missing masks in "
                      f"{miss}/{len(views)} views", flush=True)
            valid = ~np.isnan(score[:, s_idx])
            if not valid.any():
                print(f"[residue]   {slug}: 0 valid points", flush=True)
                continue
            s = score[valid, s_idx]
            print(f"[residue]   {slug}: >0.4={int((s > 0.4).sum())}  "
                  f">0.6={int((s > 0.6).sum())}  "
                  f">0.9={int((s > 0.9).sum())}  "
                  f"max={float(s.max()):.3f}", flush=True)

    return ResidueResult(
        residue_score=score,
        residue_hits=residue_hits,
        visible_views=visible_views,
        slugs=tuple(slugs),
    )


# ---------------------------------------------------------------------------
# Convenience: splat / mesh entry points
# ---------------------------------------------------------------------------

def compute_residue_scores_for_splat(
    scene_splat_path: Path,
    dataparser_json: Path,
    bounds_json: Path,
    view_source: V32ViewSource,
    sam3_cache_dir: Path,
    slugs: Sequence[str],
    **kwargs,
):
    """Load a splat PLY, inverse-transform positions to metric metres,
    call compute_residue_scores. Returns (ResidueResult, positions_metric)."""
    positions_metric = _load_splat_positions_metric(
        Path(scene_splat_path), Path(dataparser_json), Path(bounds_json))
    result = compute_residue_scores(
        positions_metric, view_source, sam3_cache_dir, slugs, **kwargs)
    return result, positions_metric


def compute_residue_scores_for_mesh(
    scene_mesh_path: Path,
    view_source: V32ViewSource,
    sam3_cache_dir: Path,
    slugs: Sequence[str],
    **kwargs,
):
    """Load a mesh PLY, use its (already-metric) vertices, call
    compute_residue_scores. Returns (ResidueResult, verts, faces)."""
    verts, faces = _load_mesh_vertices_and_faces(Path(scene_mesh_path))
    result = compute_residue_scores(
        verts, view_source, sam3_cache_dir, slugs, **kwargs)
    return result, verts, faces


# ---------------------------------------------------------------------------
# CLI stub for standalone debugging
# ---------------------------------------------------------------------------

def _parse_args():
    ap = argparse.ArgumentParser(description="SAM3 residue detector")
    ap.add_argument("--scene-splat", type=Path, default=None)
    ap.add_argument("--scene-mesh", type=Path, default=None)
    ap.add_argument("--dataparser", type=Path, required=False)
    ap.add_argument("--bounds-json", type=Path, required=False)
    ap.add_argument("--project-root", type=Path,
                    default=_PROJECT_ROOT)
    ap.add_argument("--sam3-cache", type=Path, required=True)
    ap.add_argument("--slugs", type=str, required=True,
                    help="comma-separated list of prompt slugs")
    ap.add_argument("--min-views", type=int, default=20)
    ap.add_argument("--dilate-px", type=int, default=8)
    ap.add_argument("--occlusion-mesh", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True,
                    help="output .npz path")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    slugs = [s.strip() for s in args.slugs.split(",") if s.strip()]
    vs = V32ViewSource(args.project_root)

    if args.scene_splat is not None:
        if args.dataparser is None or args.bounds_json is None:
            raise SystemExit("--scene-splat requires --dataparser + --bounds-json")
        result, positions = compute_residue_scores_for_splat(
            args.scene_splat, args.dataparser, args.bounds_json,
            vs, args.sam3_cache, slugs,
            min_views=args.min_views, dilate_px=args.dilate_px,
            occlusion_mesh_path=args.occlusion_mesh,
        )
    elif args.scene_mesh is not None:
        result, verts, faces = compute_residue_scores_for_mesh(
            args.scene_mesh, vs, args.sam3_cache, slugs,
            min_views=args.min_views, dilate_px=args.dilate_px,
            occlusion_mesh_path=args.occlusion_mesh,
        )
    else:
        raise SystemExit("provide either --scene-splat or --scene-mesh")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        residue_score=result.residue_score,
        residue_hits=result.residue_hits,
        visible_views=result.visible_views,
        slugs=np.array(list(result.slugs), dtype=object),
    )
    print(f"[residue] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
