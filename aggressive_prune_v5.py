"""
Aggressive splat pruner v5 — MESH-DISTANCE shape mask.

v1/v2/v3 used a bounding box (AABB) — either too loose, or cut the back
wall. v4 used self-density (Open3D statistical/radius outlier) — cut
into scene + still left some fringe. The user's insight: the scene MESH
itself is the right shape mask. The mesh contains only real triangulated
surfaces. Floaters are by definition not on any real surface, so they
sit FAR from any mesh face. Keep Gaussians within N cm of the mesh,
drop everything else.

Filter pipeline (v5):

  1. **Scale percentile** — same as v1-v4. Drops Gaussians whose
     max-scale axis is above the chosen percentile (default 95).

  2. **Opacity threshold** — same as v1-v4. Drops sigmoid(opacity) <
     threshold (default 0.30).

  3. **Mesh-distance shape mask** (NEW) — invert each surviving splat
     position back to metric world via the nerfstudio dataparser
     transform, then query unsigned distance to nearest mesh face via
     an Open3D RaycastingScene BVH. Keep Gaussians whose distance is
     <= --max-dist-m (default 0.03 m = 3 cm).

Usage:
    python aggressive_prune_v5.py output/splat_v32_data3_noinit_pruned.ply \
        [--out output/splat_v32_v5.ply] \
        [--scale-pct 95] [--opacity-min 0.30] \
        [--scene-mesh output/mesh_v32_data3/mesh_v32_data3_openmvs.ply] \
        [--dataparser nerfstudio/dense/splatfacto/2026-06-07_203807/dataparser_transforms.json] \
        [--bounds-json colmap/dense/tof_bounds.json] \
        [--max-dist-m 0.03] [--no-distance]

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
                   help="Output .ply. Default: <input>_v5.ply")
    p.add_argument("--scale-pct", type=float, default=95.0,
                   help="Drop Gaussians whose max-scale is above this "
                        "percentile of the population (default 95)")
    p.add_argument("--opacity-min", type=float, default=0.30,
                   help="Drop Gaussians with sigmoid(opacity) < this "
                        "(default 0.30)")
    p.add_argument("--scene-mesh", type=Path,
                   default=Path("output/mesh_v32_data3/mesh_v32_data3_openmvs.ply"),
                   help="Triangulated scene mesh used as the shape mask "
                        "(metric world units)")
    p.add_argument("--dataparser", type=Path,
                   default=Path("nerfstudio/dense/splatfacto/2026-06-07_203807/"
                                "dataparser_transforms.json"),
                   help="nerfstudio dataparser_transforms.json next to the "
                        "splatfacto run's config.yml")
    p.add_argument("--bounds-json", type=Path,
                   default=Path("colmap/dense/tof_bounds.json"),
                   help="tof_bounds.json containing scale_factor_da3_to_colmap")
    p.add_argument("--max-dist-m", type=float, default=0.03,
                   help="Keep Gaussians within this distance (metres) of the "
                        "nearest mesh face (default 0.03 = 3 cm)")
    p.add_argument("--no-distance", action="store_true",
                   help="Skip the mesh-distance filter (only run scale + "
                        "opacity filters, same as v1)")
    return p.parse_args()


def _build_raycasting_scene_from_mesh(mesh_path: Path):
    """Load a triangulated mesh via trimesh and wrap it in an Open3D
    RaycastingScene. Returns (scene, n_verts, n_faces, build_seconds).
    The scene supports compute_distance(pts) -> unsigned distance to the
    nearest triangle in the BVH.
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


def main():
    args = _parse_args()
    in_path = args.input_ply.resolve()
    out_path = (args.out or in_path.with_name(in_path.stem + "_v5.ply")).resolve()
    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        return 2

    print(f"=== aggressive_prune_v5 (mesh-distance shape mask) ===")
    print(f"  input       : {in_path}")
    print(f"  output      : {out_path}")
    print(f"  scene-mesh  : {args.scene_mesh}")
    print(f"  dataparser  : {args.dataparser}")
    print(f"  bounds-json : {args.bounds_json}")
    print(f"  max-dist    : {args.max_dist_m:.3f} m "
          f"({args.max_dist_m*100:.1f} cm)")
    if args.no_distance:
        print(f"  (--no-distance set: mesh-distance filter DISABLED)")

    # ── Read input splat PLY (reuse v1 binary reader) ─────────────────
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

    # ── Filter 1: scale percentile ────────────────────────────────────
    scale_thresh = float(np.percentile(max_scale, args.scale_pct))
    scale_mask = max_scale <= scale_thresh
    dropped = int((~scale_mask).sum())
    print(f"\n  filter 1 — scale > p{args.scale_pct} ({scale_thresh:.4f}): "
          f"drop {dropped:,} Gaussians ({100*dropped/n:.1f}%)")
    keep &= scale_mask

    # ── Filter 2: opacity threshold ───────────────────────────────────
    op_mask = op >= args.opacity_min
    dropped = int((~op_mask & keep).sum())
    print(f"  filter 2 — opacity < {args.opacity_min}: drop additional "
          f"{dropped:,} Gaussians")
    keep &= op_mask

    n_pre_distance = int(keep.sum())
    print(f"  after scale + opacity: {n_pre_distance:,} / {n:,} "
          f"({100*n_pre_distance/n:.1f}%)")

    # ── Filter 3: mesh-distance shape mask ────────────────────────────
    if not args.no_distance:
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

        # Invert SURVIVING splat positions to metric world.
        survivor_idx = np.where(keep)[0]
        pos_splat = arr[survivor_idx][:,
                       [name_to_idx["x"], name_to_idx["y"], name_to_idx["z"]]
                   ].astype(np.float64)
        pos_metric = _splat_to_metric(pos_splat, R, t, dp_scale, colmap_to_metric)
        print(f"  inverted {len(pos_metric):,} splat positions to "
              f"metric world")
        print(f"    metric BB: "
              f"x[{pos_metric[:,0].min():.2f},{pos_metric[:,0].max():.2f}]  "
              f"y[{pos_metric[:,1].min():.2f},{pos_metric[:,1].max():.2f}]  "
              f"z[{pos_metric[:,2].min():.2f},{pos_metric[:,2].max():.2f}]")

        # Distance query.
        import open3d as o3d
        t0 = time.perf_counter()
        query = o3d.core.Tensor(pos_metric.astype(np.float32))
        distances = scene.compute_distance(query).numpy()
        t_query = time.perf_counter() - t0
        print(f"  distance query: {t_query:.2f} s "
              f"({len(distances):,} points)")

        # Distance distribution.
        med = float(np.median(distances))
        p95 = float(np.percentile(distances, 95))
        dmax = float(distances.max())
        print(f"    distance to nearest face: median {med*100:.2f} cm  "
              f"p95 {p95*100:.2f} cm  max {dmax*100:.2f} cm")

        # Apply distance mask.
        dist_mask_survivors = distances <= args.max_dist_m
        n_kept_distance = int(dist_mask_survivors.sum())
        n_dropped_distance = int((~dist_mask_survivors).sum())
        pct_drop = 100.0 * n_dropped_distance / max(n_pre_distance, 1)
        print(f"\n  filter 3 — distance > {args.max_dist_m*100:.1f} cm "
              f"from mesh:")
        print(f"    n_pre_distance      = {n_pre_distance:,}")
        print(f"    n_kept_distance     = {n_kept_distance:,}")
        print(f"    n_dropped_distance  = {n_dropped_distance:,} "
              f"({pct_drop:.1f}%)")

        # Project survivor-mask back onto full keep array.
        full_dist_mask = np.zeros(n, dtype=bool)
        full_dist_mask[survivor_idx] = dist_mask_survivors
        keep &= full_dist_mask

    # ── Write output ──────────────────────────────────────────────────
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
