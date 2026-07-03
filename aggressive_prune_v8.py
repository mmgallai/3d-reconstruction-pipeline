"""
Aggressive splat pruner v8 — RENDER-VIEW UTILIZATION filter.

Gemini's Phase-3 / Alternative-1 recommendation: detect view-dependent
specular blobs and floaters by measuring how many of the 140 training
views can ACTUALLY observe each Gaussian's centre. View-dependent
specular blobs are visible from only a narrow range of incidence
angles; transient floaters land in front of the real surface in just a
few outlier views. Genuine, surface-anchored Gaussians, by contrast,
are visible from many views. Drop the low-utilization tail.

Practical simplification used here (escalate to true alpha-rendering
contribution if needed):

  For each surviving Gaussian centre:
    For each of the 140 training views:
      1. Project the centre to the camera (using w2c).
      2. in_front  -> z_cam > 5 cm
      3. in_frame  -> 0 <= u < W AND 0 <= v < H (using K)
      4. Cast a ray from cam_pos through the centre. Compare its
         `t_hit` to ||pos - cam_pos||. Unobstructed iff
         t_hit >= dist - depth_tol.
    utilization = count of views with in_front & in_frame & unobstructed.

Then drop Gaussians with utilization < --min-views.

This is a CONSERVATIVE proxy. It catches:
  * floaters whose centres land behind real geometry in most views
  * sparse mid-air floaters visible only from edge cases
It misses view-dependent blobs whose centres ARE geometrically visible
from many views but only CONTRIBUTE COLOUR to a few -- those need
the full alpha-rendering test.

Same pattern as `_unseen_core_map.py`: build a RaycastingScene from
the scene mesh, iterate over views, cast a per-point ray, compare
hit_t to dist - tol.

Filter pipeline (v8):
  1. **Scale percentile** -- same as v1-v7 (default p95).
  2. **Opacity threshold** -- same as v1-v7 (default 0.30).
  3. **Render-view utilization** (NEW) -- drop Gaussians whose centre
     is potentially visible from < --min-views training views.

Usage:
    python aggressive_prune_v8.py output/splat_v32_data3_noinit_pruned.ply \
        --out output/splat_cleanup_phases/phase3_render_utilization/<name>.ply \
        --scene-mesh output/mesh_v32_data3/mesh_v32_data3_openmvs.ply \
        --dataparser nerfstudio/dense/splatfacto/2026-06-07_203807/dataparser_transforms.json \
        --bounds-json colmap/dense/tof_bounds.json \
        --project-root . \
        [--scale-pct 95] [--opacity-min 0.30] \
        [--min-views 10] [--depth-tol-m 0.02] [--no-utilization]

The script does NOT touch the input file. A fresh .ply is written.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Reuse v1 IO + v32 dataparser inversion helpers.
from aggressive_prune import _read_ply, _write_ply
from _spatial_crop_splat import _load_dataparser_transform, _splat_to_metric


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_ply", type=Path)
    p.add_argument("--out", type=Path, default=None,
                   help="Output .ply. Default: <input>_cleaned_v8.ply")
    p.add_argument("--scale-pct", type=float, default=95.0,
                   help="Drop Gaussians whose max-scale is above this "
                        "percentile of the population (default 95)")
    p.add_argument("--opacity-min", type=float, default=0.30,
                   help="Drop Gaussians with sigmoid(opacity) < this "
                        "(default 0.30)")
    p.add_argument("--scene-mesh", type=Path,
                   default=Path("output/mesh_v32_data3/mesh_v32_data3_openmvs.ply"),
                   help="Triangulated scene mesh for the occlusion test "
                        "(metric world units)")
    p.add_argument("--dataparser", type=Path,
                   default=Path("nerfstudio/dense/splatfacto/2026-06-07_203807/"
                                "dataparser_transforms.json"),
                   help="nerfstudio dataparser_transforms.json next to the "
                        "splatfacto run's config.yml")
    p.add_argument("--bounds-json", type=Path,
                   default=Path("colmap/dense/tof_bounds.json"),
                   help="tof_bounds.json containing scale_factor_da3_to_colmap")
    p.add_argument("--project-root", type=Path, default=Path("."),
                   help="Project root passed to V32ViewSource (so it can "
                        "find nerfstudio_data/, colmap/, etc.)")
    p.add_argument("--min-views", type=int, default=10,
                   help="Drop Gaussians visible from fewer than this many "
                        "training views (default 10; ~7%% of 140 views)")
    p.add_argument("--depth-tol-m", type=float, default=0.02,
                   help="Occlusion slack in metres: a view is 'unobstructed' "
                        "if the scene-mesh ray hits at distance >= "
                        "|pos - cam| - depth_tol_m (default 0.02 = 2 cm)")
    p.add_argument("--no-utilization", action="store_true",
                   help="Skip the render-view utilization filter (only run "
                        "scale + opacity filters)")
    return p.parse_args()


def _build_raycasting_scene_from_mesh(mesh_path: Path):
    """Same pattern as aggressive_prune_v5 + _unseen_core_map: load a
    triangulated mesh via trimesh and wrap it in an Open3D RaycastingScene.
    Returns (scene, n_verts, n_faces, t_load, t_bvh).
    """
    import open3d as o3d
    import trimesh

    t0 = time.perf_counter()
    m = trimesh.load(str(mesh_path), force="mesh", process=False)
    if m.is_empty or len(m.faces) == 0:
        raise RuntimeError(f"Scene mesh has no faces: {mesh_path}")
    verts = np.asarray(m.vertices, dtype=np.float32)
    faces = np.asarray(m.faces, dtype=np.uint32)
    t_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    tmesh = o3d.t.geometry.TriangleMesh()
    tmesh.vertex.positions = o3d.core.Tensor(verts)
    tmesh.triangle.indices = o3d.core.Tensor(faces)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)
    t_bvh = time.perf_counter() - t0

    return scene, len(verts), len(faces), t_load, t_bvh


def _count_views_per_point(scene, points_metric: np.ndarray, views,
                            depth_tol_m: float) -> np.ndarray:
    """For each metric 3D point, count the number of training views in
    which the point is in front of the camera, projects inside the image
    frame, AND is unobstructed by the scene mesh.

    Mirrors `_count_views_per_point` in `_unseen_core_map.py`: same
    project + occlusion test, batched over all points per view.
    """
    import open3d as o3d
    n_pts = len(points_metric)
    n_views = len(views)
    seen = np.zeros(n_pts, dtype=np.int32)
    print(f"  utilization: casting {n_pts:,} points x {n_views} views = "
          f"{n_pts * n_views:,} rays total")
    pts32 = points_metric.astype(np.float32)
    for vi, view in enumerate(views):
        cam_pos = view.c2w[:3, 3].astype(np.float32)
        # Vector from camera to point + its length.
        dirs = pts32 - cam_pos[None, :]
        dists = np.linalg.norm(dirs, axis=-1)
        valid_dist = dists > 0.05

        # In-frame test (use w2c + K, same as _unseen_core_map).
        w2c = view.w2c
        K = view.K
        pts_cam = (w2c[:3, :3] @ pts32.T).T + w2c[:3, 3]
        zc = pts_cam[:, 2]
        in_front = zc > 0.05
        safe_zc = np.where(in_front, zc, 1.0)
        u = K[0, 0] * pts_cam[:, 0] / safe_zc + K[0, 2]
        v = K[1, 1] * pts_cam[:, 1] / safe_zc + K[1, 2]
        in_frame = (u >= 0) & (u < view.width) & (v >= 0) & (v < view.height)

        # Occlusion test: cast a ray from the camera through the point and
        # compare the scene-mesh hit distance to ||pos - cam||.
        dirs_norm = dirs / np.maximum(dists[:, None], 1e-9)
        origins = np.broadcast_to(cam_pos, dirs_norm.shape)
        rays_np = np.concatenate([origins, dirs_norm], axis=-1).astype(np.float32)
        ans = scene.cast_rays(o3d.core.Tensor(rays_np))
        hit_t = ans["t_hit"].numpy()
        unobstructed = hit_t >= (dists - depth_tol_m)

        visible = valid_dist & in_front & in_frame & unobstructed
        seen[visible] += 1
        if (vi + 1) % 20 == 0 or vi == n_views - 1:
            print(f"  utilization: view {vi+1}/{n_views}  "
                  f"visible-now={int(visible.sum()):>6}  "
                  f"cumulative >=1 view: "
                  f"{int((seen > 0).sum()):>6}/{n_pts}")
    return seen


def main():
    args = _parse_args()
    in_path = args.input_ply.resolve()
    out_path = (args.out or in_path.with_name(in_path.stem + "_cleaned_v8.ply")).resolve()
    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        return 2

    print(f"=== aggressive_prune_v8 (render-view utilization filter) ===")
    print(f"  input        : {in_path}")
    print(f"  output       : {out_path}")
    print(f"  scene-mesh   : {args.scene_mesh}")
    print(f"  dataparser   : {args.dataparser}")
    print(f"  bounds-json  : {args.bounds_json}")
    print(f"  project-root : {args.project_root}")
    print(f"  min-views    : {args.min_views}")
    print(f"  depth-tol    : {args.depth_tol_m:.3f} m "
          f"({args.depth_tol_m*100:.1f} cm)")
    if args.no_utilization:
        print(f"  (--no-utilization set: render-view filter DISABLED)")

    # Read input splat PLY (reuse v1 binary reader).
    n, props, raw, rec_size, header = _read_ply(in_path)
    arr = np.frombuffer(raw, dtype=np.float32).reshape(n, len(props)).copy()
    name_to_idx = {p: i for i, p in enumerate(props)}
    print(f"\n  loaded {n:,} Gaussians, {len(props)} properties each")

    x = arr[:, name_to_idx["x"]]
    y = arr[:, name_to_idx["y"]]
    z = arr[:, name_to_idx["z"]]
    op_logit = arr[:, name_to_idx["opacity"]]
    op = 1.0 / (1.0 + np.exp(-op_logit))
    scale_keys = sorted([p for p in props if p.startswith("scale_")])
    scales = np.stack([arr[:, name_to_idx[k]] for k in scale_keys], axis=1)
    max_scale = np.exp(scales).max(axis=1)

    print(f"  current stats:")
    print(f"    opacity median {np.median(op):.3f}  min {op.min():.3f}  "
          f"max {op.max():.3f}")
    print(f"    max-scale median {np.median(max_scale):.4f}  "
          f"p95 {np.percentile(max_scale, 95):.4f}  "
          f"max {max_scale.max():.4f}")
    print(f"    splat BB: x[{x.min():.2f},{x.max():.2f}]  "
          f"y[{y.min():.2f},{y.max():.2f}]  "
          f"z[{z.min():.2f},{z.max():.2f}]")

    keep = np.ones(n, dtype=bool)

    # Filter 1: scale percentile.
    scale_thresh = float(np.percentile(max_scale, args.scale_pct))
    scale_mask = max_scale <= scale_thresh
    dropped_scale = int((~scale_mask).sum())
    print(f"\n  filter 1 -- scale > p{args.scale_pct} ({scale_thresh:.4f}): "
          f"drop {dropped_scale:,} Gaussians ({100*dropped_scale/n:.1f}%)")
    keep &= scale_mask

    # Filter 2: opacity threshold.
    op_mask = op >= args.opacity_min
    dropped_op = int((~op_mask & keep).sum())
    print(f"  filter 2 -- opacity < {args.opacity_min}: drop additional "
          f"{dropped_op:,} Gaussians")
    keep &= op_mask

    n_pre_util = int(keep.sum())
    print(f"  after scale + opacity: {n_pre_util:,} / {n:,} "
          f"({100*n_pre_util/n:.1f}%)")

    dropped_util = 0

    # Filter 3: render-view utilization.
    if not args.no_utilization:
        # Load + invert dataparser transform.
        dp_path = args.dataparser.resolve()
        if not dp_path.exists():
            print(f"ERROR: dataparser not found: {dp_path}", file=sys.stderr)
            return 2
        R, t, dp_scale = _load_dataparser_transform(dp_path)
        print(f"\n  dataparser: scale={dp_scale:.4f}, "
              f"R det={np.linalg.det(R):.4f}")

        # Load tof bounds -> colmap_to_metric.
        bp = args.bounds_json.resolve()
        if not bp.exists():
            print(f"ERROR: bounds-json not found: {bp}", file=sys.stderr)
            return 2
        bj = json.loads(bp.read_text())
        scale_factor = float(bj["scale_factor_da3_to_colmap"])
        colmap_to_metric = 1.0 / scale_factor
        print(f"  scale_factor_da3_to_colmap = {scale_factor:.6f}  "
              f"(colmap_to_metric = {colmap_to_metric:.6f})")

        # Build raycasting scene over the mesh.
        mp = args.scene_mesh.resolve()
        if not mp.exists():
            print(f"ERROR: scene mesh not found: {mp}", file=sys.stderr)
            return 2
        print(f"  loading scene mesh + building BVH ...")
        scene, n_v, n_f, t_load, t_bvh = _build_raycasting_scene_from_mesh(mp)
        print(f"    mesh load: {t_load:.2f} s  ({n_v:,} verts, "
              f"{n_f:,} faces)")
        print(f"    BVH build: {t_bvh:.2f} s")

        # Load V32 views (140 metric poses).
        sys.path.insert(0, str(Path(__file__).parent.resolve()))
        from scene_segmenter.views import V32ViewSource
        print(f"  loading V32 view source from {args.project_root} ...")
        t0 = time.perf_counter()
        vs = V32ViewSource(args.project_root,
                            bounds_json=args.bounds_json)
        views = vs.all_views()
        t_views = time.perf_counter() - t0
        print(f"    loaded {len(views)} views in {t_views:.2f} s")

        # Invert SURVIVING splat positions to metric world.
        survivor_idx = np.where(keep)[0]
        pos_splat = arr[survivor_idx][:,
                       [name_to_idx["x"], name_to_idx["y"], name_to_idx["z"]]
                   ].astype(np.float64)
        pos_metric = _splat_to_metric(pos_splat, R, t, dp_scale,
                                       colmap_to_metric)
        print(f"  inverted {len(pos_metric):,} splat positions to "
              f"metric world")
        print(f"    metric BB: "
              f"x[{pos_metric[:,0].min():.2f},{pos_metric[:,0].max():.2f}]  "
              f"y[{pos_metric[:,1].min():.2f},{pos_metric[:,1].max():.2f}]  "
              f"z[{pos_metric[:,2].min():.2f},{pos_metric[:,2].max():.2f}]")

        # Count visible views per Gaussian centre.
        t0 = time.perf_counter()
        utilization = _count_views_per_point(scene, pos_metric, views,
                                              args.depth_tol_m)
        t_util = time.perf_counter() - t0
        print(f"\n  utilization computed in {t_util:.2f} s "
              f"({len(pos_metric):,} pts x {len(views)} views)")

        # Distribution of utilization.
        n_views = len(views)
        med = float(np.median(utilization))
        p10 = float(np.percentile(utilization, 10))
        p25 = float(np.percentile(utilization, 25))
        p50 = float(np.percentile(utilization, 50))
        p90 = float(np.percentile(utilization, 90))
        umax = int(utilization.max())
        umin = int(utilization.min())
        print(f"  utilization distribution (out of {n_views} views):")
        print(f"    min  = {umin}   p10 = {p10:.0f}   p25 = {p25:.0f}   "
              f"median = {p50:.0f}   p90 = {p90:.0f}   max = {umax}")
        n_zero = int((utilization == 0).sum())
        n_lt_min = int((utilization < args.min_views).sum())
        print(f"    n with 0 views: {n_zero:,} "
              f"({100*n_zero/max(len(utilization),1):.1f}%)")
        print(f"    n with < {args.min_views} views: {n_lt_min:,} "
              f"({100*n_lt_min/max(len(utilization),1):.1f}%)")

        # Apply utilization mask.
        util_mask_survivors = utilization >= args.min_views
        n_kept_util = int(util_mask_survivors.sum())
        dropped_util = int((~util_mask_survivors).sum())
        pct_drop_util = 100.0 * dropped_util / max(n_pre_util, 1)
        print(f"\n  filter 3 -- utilization < {args.min_views} views:")
        print(f"    n_pre_utilization     = {n_pre_util:,}")
        print(f"    n_kept_utilization    = {n_kept_util:,}")
        print(f"    n_dropped_utilization = {dropped_util:,} "
              f"({pct_drop_util:.1f}% of survivors)")

        # Project survivor-mask back onto full keep array.
        full_util_mask = np.zeros(n, dtype=bool)
        full_util_mask[survivor_idx] = util_mask_survivors
        keep &= full_util_mask

    # Per-filter drop summary.
    print(f"\n  per-filter drop counts:")
    print(f"    filter 1 (scale > p{args.scale_pct}): {dropped_scale:,}")
    print(f"    filter 2 (opacity < {args.opacity_min}): {dropped_op:,}")
    print(f"    filter 3 (utilization < {args.min_views}): {dropped_util:,}")

    # Write output.
    kept = arr[keep]
    print(f"\n  kept {int(keep.sum()):,} / {n:,} Gaussians "
          f"({100*int(keep.sum())/n:.1f}%)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_ply(out_path, header, kept)
    size_mb = out_path.stat().st_size / 1_048_576
    print(f"  saved: {out_path}  ({size_mb:.1f} MB)")
    print(f"\n=== Done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
