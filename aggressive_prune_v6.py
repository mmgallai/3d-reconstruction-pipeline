"""
Aggressive splat pruner v6 — SHAPE-AWARE mesh-distance filters.

Motivation
----------
v5 dropped Gaussians whose CENTER was > N cm from the nearest mesh face.
That works for floaters, but the user reports the d25cm output still has
leafy/needle artifacts. Inspection of survivors shows:

    median max-axis size:   0.87 cm metric
    median anisotropy:     12.7x
    p90 anisotropy:        245x
    12.6% of survivors are >2 cm long AND >5x anisotropic

These are "needle" Gaussians whose CENTER sits within max_dist_m of the
mesh (so v5 keeps them), but whose long axis extends well outside the
mesh — that's exactly the leafy artifact pattern.

v6 adds SHAPE-AWARE filter modes that look at each Gaussian's longest
axis (max(exp(scale_*))) and/or its anisotropy ratio.

Filter pipeline (v6)
--------------------

  1. Scale percentile  (same as v1-v5)
  2. Opacity threshold (same as v1-v5)
  3. Shape filter — one of:

       mesh-dist   keep if center_dist_m <= max_dist_m
                   (identical to v5)

       eff-extent  keep if (center_dist_m + alpha * max_axis_m) <= max_dist_m
                   (account for the longest semi-axis — drops needles that
                    poke out past the mesh)  [DEFAULT]

       aniso       keep if (max_axis / min_axis) <= max_aniso
                   (drop any Gaussian more anisotropic than this ratio;
                    no mesh distance check)

       combo       keep if all of:
                       center_dist_m                           <= max_dist_m
                       center_dist_m + alpha * max_axis_m      <= max_dist_m
                       max_axis / min_axis                     <= max_aniso

       off         no shape filter (only scale + opacity)

Geometry / units
----------------
scale_0..2 in the PLY are log-space; the metric semi-axis lengths are
exp(scale_i). The V32 splat-per-metre factor is 1.5227, so:

    max_axis_metric = max(exp(scale_*)) / 1.5227
    min_axis_metric = min(exp(scale_*)) / 1.5227
    aniso           = max(exp(scale_*)) / min(exp(scale_*))   (ratio is unitless)

Usage
-----
    python aggressive_prune_v6.py output/splat_v32_data3_noinit_pruned.ply \
        [--out output/splat_v32_v6.ply] \
        [--scale-pct 95] [--opacity-min 0.30] \
        [--scene-mesh output/mesh_v32_data3/mesh_v32_data3_openmvs.ply] \
        [--dataparser nerfstudio/dense/splatfacto/2026-06-07_203807/dataparser_transforms.json] \
        [--bounds-json colmap/dense/tof_bounds.json] \
        [--filter eff-extent] \
        [--max-dist-m 0.03] [--extent-alpha 1.0] [--max-aniso 50.0]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Reuse v1 IO + v32 dataparser inversion helpers (same as v5).
from aggressive_prune import _read_ply, _write_ply
from _spatial_crop_splat import _load_dataparser_transform, _splat_to_metric


# V32 splat-per-metre factor (same units convention used elsewhere in repo).
SPLAT_PER_METRE = 1.5227


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_ply", type=Path)
    p.add_argument("--out", type=Path, default=None,
                   help="Output .ply. Default: <input>_v6.ply")
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
    p.add_argument("--filter",
                   choices=["mesh-dist", "eff-extent", "aniso", "combo", "off"],
                   default="eff-extent",
                   help="Shape filter mode (default 'eff-extent'). See module "
                        "docstring for details.")
    p.add_argument("--max-dist-m", type=float, default=0.03,
                   help="Keep Gaussians within this metric distance of the "
                        "mesh (default 0.03 = 3 cm). Used by mesh-dist, "
                        "eff-extent, and combo.")
    p.add_argument("--extent-alpha", type=float, default=1.0,
                   help="Multiplier on max_axis_metric for eff-extent and "
                        "combo modes (default 1.0). Larger alpha = stricter "
                        "needle rejection.")
    p.add_argument("--max-aniso", type=float, default=50.0,
                   help="Maximum allowed max-axis/min-axis ratio for aniso "
                        "and combo modes (default 50). Anything more "
                        "anisotropic is dropped.")
    return p.parse_args()


def _build_raycasting_scene_from_mesh(mesh_path: Path):
    """Load a triangulated mesh via trimesh and wrap it in an Open3D
    RaycastingScene. Returns (scene, n_verts, n_faces, t_load, t_bvh).
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
    out_path = (args.out or in_path.with_name(in_path.stem + "_v6.ply")).resolve()
    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        return 2

    print(f"=== aggressive_prune_v6 (shape-aware mesh filter) ===")
    print(f"  input         : {in_path}")
    print(f"  output        : {out_path}")
    print(f"  scene-mesh    : {args.scene_mesh}")
    print(f"  dataparser    : {args.dataparser}")
    print(f"  bounds-json   : {args.bounds_json}")
    print(f"  filter mode   : {args.filter}")
    print(f"  max-dist      : {args.max_dist_m:.3f} m "
          f"({args.max_dist_m*100:.1f} cm)")
    print(f"  extent-alpha  : {args.extent_alpha}")
    print(f"  max-aniso     : {args.max_aniso}")

    # Whether we need mesh-distance at all (eff-extent / mesh-dist / combo)
    need_distance = args.filter in ("mesh-dist", "eff-extent", "combo")
    # Whether we need anisotropy (aniso / combo)
    need_aniso = args.filter in ("aniso", "combo")
    # Whether we need the max-axis size (eff-extent / combo)
    need_extent = args.filter in ("eff-extent", "combo")

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
    # Actual semi-axis lengths in SPLAT units.
    actual_scales = np.exp(scales)
    max_scale = actual_scales.max(axis=1)
    min_scale = actual_scales.min(axis=1)
    # Anisotropy ratio (unitless, identical in splat and metric space).
    aniso_full = max_scale / np.maximum(min_scale, 1e-12)

    print(f"  current stats:")
    print(f"    opacity median {np.median(op):.3f}  min {op.min():.3f}  "
          f"max {op.max():.3f}")
    print(f"    max-scale (splat) median {np.median(max_scale):.4f}  "
          f"p95 {np.percentile(max_scale, 95):.4f}  "
          f"max {max_scale.max():.4f}")
    print(f"    aniso     median {np.median(aniso_full):.2f}x  "
          f"p95 {np.percentile(aniso_full, 95):.2f}x  "
          f"max {aniso_full.max():.2f}x")
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

    n_pre_shape = int(keep.sum())
    print(f"  after scale + opacity: {n_pre_shape:,} / {n:,} "
          f"({100*n_pre_shape/n:.1f}%)")

    # ── Filter 3: shape-aware filter ──────────────────────────────────
    if args.filter == "off":
        print(f"\n  filter 3 — DISABLED (--filter off)")
    else:
        survivor_idx = np.where(keep)[0]

        # Per-survivor anisotropy + metric extent (always cheap to compute).
        max_scale_surv = max_scale[survivor_idx]
        min_scale_surv = min_scale[survivor_idx]
        max_axis_m = max_scale_surv / SPLAT_PER_METRE
        aniso_surv = max_scale_surv / np.maximum(min_scale_surv, 1e-12)

        # Distance query (only if filter needs it).
        center_dist_m = None
        if need_distance:
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

            import open3d as o3d
            t0 = time.perf_counter()
            query = o3d.core.Tensor(pos_metric.astype(np.float32))
            distances = scene.compute_distance(query).numpy()
            t_query = time.perf_counter() - t0
            print(f"  distance query: {t_query:.2f} s "
                  f"({len(distances):,} points)")

            center_dist_m = distances.astype(np.float64)

            med = float(np.median(center_dist_m))
            p95 = float(np.percentile(center_dist_m, 95))
            dmax = float(center_dist_m.max())
            print(f"    center distance to nearest face: "
                  f"median {med*100:.2f} cm  p95 {p95*100:.2f} cm  "
                  f"max {dmax*100:.2f} cm")
            if need_extent:
                eff = center_dist_m + args.extent_alpha * max_axis_m
                print(f"    effective extent (center + {args.extent_alpha}*"
                      f"max_axis): median {np.median(eff)*100:.2f} cm  "
                      f"p95 {np.percentile(eff, 95)*100:.2f} cm  "
                      f"max {eff.max()*100:.2f} cm")
            print(f"    max-axis metric: median "
                  f"{np.median(max_axis_m)*100:.2f} cm  "
                  f"p95 {np.percentile(max_axis_m, 95)*100:.2f} cm  "
                  f"max {max_axis_m.max()*100:.2f} cm")

        # Build per-criterion masks (over the SURVIVOR subset).
        n_surv = len(survivor_idx)
        crit_masks = {}
        if need_distance:
            crit_masks["center_dist <= max_dist"] = (
                center_dist_m <= args.max_dist_m
            )
        if need_extent:
            crit_masks[
                f"center_dist + {args.extent_alpha}*max_axis <= max_dist"
            ] = ((center_dist_m + args.extent_alpha * max_axis_m)
                 <= args.max_dist_m)
        if need_aniso:
            crit_masks["aniso <= max_aniso"] = aniso_surv <= args.max_aniso

        print(f"\n  filter 3 — '{args.filter}' criteria "
              f"(n_pre_shape = {n_surv:,}):")
        for name, m in crit_masks.items():
            n_pass = int(m.sum())
            n_fail = int((~m).sum())
            pct_fail = 100.0 * n_fail / max(n_surv, 1)
            print(f"    [{name}] pass {n_pass:,}  fail {n_fail:,} "
                  f"({pct_fail:.1f}%)")

        # Combine according to mode.
        if args.filter == "mesh-dist":
            shape_mask = crit_masks["center_dist <= max_dist"]
        elif args.filter == "eff-extent":
            shape_mask = crit_masks[
                f"center_dist + {args.extent_alpha}*max_axis <= max_dist"
            ]
        elif args.filter == "aniso":
            shape_mask = crit_masks["aniso <= max_aniso"]
        elif args.filter == "combo":
            shape_mask = np.ones(n_surv, dtype=bool)
            for m in crit_masks.values():
                shape_mask &= m
        else:
            raise AssertionError(f"unreachable filter mode {args.filter}")

        n_kept_shape = int(shape_mask.sum())
        n_dropped_shape = int((~shape_mask).sum())
        pct_drop = 100.0 * n_dropped_shape / max(n_surv, 1)
        print(f"    COMBINED ({args.filter}): keep {n_kept_shape:,}  "
              f"drop {n_dropped_shape:,} ({pct_drop:.1f}%)")

        # Project survivor-mask back onto full keep array.
        full_shape_mask = np.zeros(n, dtype=bool)
        full_shape_mask[survivor_idx] = shape_mask
        keep &= full_shape_mask

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
