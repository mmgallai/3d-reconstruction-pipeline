"""
Aggressive splat pruner v9 — INTERSECTION of Phase 2 (v7 surface-normal
projected covariance) and Phase 3 (v8 render-view utilization).

Gemini's Phase 4 recommendation: compose v7 + v8 via INTERSECTION of keep
masks rather than using either standalone. The surface-normal filter (v7)
already removes the off-surface floaters that drive low utilization
scores, so the intersection should retain ~40-55 percent of the original
splat while specifically targeting view-dependent reflective-monitor
blobs.

A Gaussian survives v9 (intersection mode) iff:
    (1) sigma_perp <= --max-perp-m   (geometrically on-surface in shape)
  AND
    (2) utilization >= --min-views   (visible from enough cameras)

Drops have the best rationale of any phase so far:
  * geometrically off-surface (V_perp fails) AND
  * not seen by enough cameras (utilization fails) -> almost certainly
    a blob/floater/needle, not real scene content.

Filter pipeline (v9):
  1. **Scale percentile**      -- same as v1-v8 (default p95).
  2. **Opacity threshold**     -- same as v1-v8 (default 0.30).
  3. **Surface-normal V_perp** -- v7 logic.
  4. **View utilization**      -- v8 logic.
  5. **Combine** (3) and (4) per --mode (intersection or union).

Usage:
    python aggressive_prune_v9.py output/splat_v32_data3_noinit_pruned.ply \
        --out output/splat_cleanup_phases/phase4_intersection/splat_v9.ply \
        --scene-mesh output/mesh_v32_data3/mesh_v32_data3_openmvs.ply \
        --dataparser nerfstudio/dense/splatfacto/2026-06-07_203807/dataparser_transforms.json \
        --bounds-json colmap/dense/tof_bounds.json \
        --project-root . \
        [--scale-pct 95] [--opacity-min 0.30] \
        [--max-perp-m 0.01] [--min-views 5] [--depth-tol-m 0.02] \
        [--mode intersection|union] \
        [--no-shape-filter] [--no-utilization]

The script does NOT touch the input file. A fresh .ply is written.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Reuse v1 IO + dataparser inversion helpers.
from aggressive_prune import _read_ply, _write_ply
from _spatial_crop_splat import _load_dataparser_transform, _splat_to_metric

# Reuse v7 helpers (V_perp covariance projection + raycasting scene + normals).
from aggressive_prune_v7 import (
    _build_raycasting_scene_and_normals,
    _quat_to_rot_matrix,
    _projected_perp_sigma,
)

# Reuse v8 helper (per-point view-utilization counter).
from aggressive_prune_v8 import _count_views_per_point


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_ply", type=Path)
    p.add_argument("--out", type=Path, default=None,
                   help="Output .ply. Default: <input>_cleaned_v9.ply")
    p.add_argument("--scale-pct", type=float, default=95.0,
                   help="Drop Gaussians whose max-scale is above this "
                        "percentile of the population (default 95)")
    p.add_argument("--opacity-min", type=float, default=0.30,
                   help="Drop Gaussians with sigmoid(opacity) < this "
                        "(default 0.30)")
    p.add_argument("--scene-mesh", type=Path,
                   default=Path("output/mesh_v32_data3/mesh_v32_data3_openmvs.ply"),
                   help="Triangulated scene mesh used for BOTH the surface "
                        "normals (Phase 2) and the occlusion test (Phase 3); "
                        "must be in metric world units")
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
    p.add_argument("--max-perp-m", type=float, default=0.01,
                   help="KEEP Gaussians whose perpendicular std-dev "
                        "(sigma_perp = sqrt(N^T Sigma_metric N)) is <= this "
                        "in metres. Default 0.01 = 1 cm (v7 production pick).")
    p.add_argument("--min-views", type=int, default=5,
                   help="KEEP Gaussians visible from at least this many "
                        "training views (default 5; Phase 3's least-bad "
                        "threshold)")
    p.add_argument("--depth-tol-m", type=float, default=0.02,
                   help="Occlusion slack in metres: a view is 'unobstructed' "
                        "if the scene-mesh ray hits at distance >= "
                        "|pos - cam| - depth_tol_m (default 0.02 = 2 cm)")
    p.add_argument("--mode", choices=["intersection", "union"],
                   default="intersection",
                   help="How to combine the V_perp keep mask and the "
                        "utilization keep mask (default intersection)")
    p.add_argument("--no-shape-filter", action="store_true",
                   help="Skip the surface-normal projected-covariance filter "
                        "(only run scale + opacity + utilization).")
    p.add_argument("--no-utilization", action="store_true",
                   help="Skip the render-view utilization filter "
                        "(only run scale + opacity + V_perp).")
    return p.parse_args()


def main():
    args = _parse_args()
    in_path = args.input_ply.resolve()
    out_path = (args.out or in_path.with_name(in_path.stem + "_cleaned_v9.ply")).resolve()
    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        return 2

    print(f"=== aggressive_prune_v9 (Phase 2 INTERSECT Phase 3) ===")
    print(f"  input        : {in_path}")
    print(f"  output       : {out_path}")
    print(f"  scene-mesh   : {args.scene_mesh}")
    print(f"  dataparser   : {args.dataparser}")
    print(f"  bounds-json  : {args.bounds_json}")
    print(f"  project-root : {args.project_root}")
    print(f"  mode         : {args.mode}")
    print(f"  max-perp     : {args.max_perp_m:.4f} m "
          f"({args.max_perp_m*100:.2f} cm)")
    print(f"  min-views    : {args.min_views}")
    print(f"  depth-tol    : {args.depth_tol_m:.3f} m "
          f"({args.depth_tol_m*100:.1f} cm)")
    if args.no_shape_filter:
        print(f"  (--no-shape-filter set: V_perp filter DISABLED)")
    if args.no_utilization:
        print(f"  (--no-utilization set: utilization filter DISABLED)")
    if args.no_shape_filter and args.no_utilization:
        print(f"  WARN: both V_perp and utilization disabled; this is "
              f"identical to aggressive_prune.py")

    # Read input splat PLY.
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
    scales_log = np.stack([arr[:, name_to_idx[k]] for k in scale_keys], axis=1)
    scales_splat_all = np.exp(scales_log)                            # (N, 3) std-dev in splat units
    max_scale = scales_splat_all.max(axis=1)

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
    dropped_scale = int((~scale_mask).sum())
    print(f"\n  filter 1 — scale > p{args.scale_pct} ({scale_thresh:.4f}): "
          f"drop {dropped_scale:,} Gaussians ({100*dropped_scale/n:.1f}%)")
    keep &= scale_mask

    # ── Filter 2: opacity threshold ───────────────────────────────────
    op_mask = op >= args.opacity_min
    dropped_op = int((~op_mask & keep).sum())
    print(f"  filter 2 — opacity < {args.opacity_min}: drop additional "
          f"{dropped_op:,} Gaussians")
    keep &= op_mask

    n_pre_phase34 = int(keep.sum())
    print(f"  after scale + opacity: {n_pre_phase34:,} / {n:,} "
          f"({100*n_pre_phase34/n:.1f}%)")

    if n_pre_phase34 == 0:
        print(f"  WARN: nothing survived scale + opacity; nothing to combine.")
        kept = arr[keep]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _write_ply(out_path, header, kept)
        return 0

    # Survivors of (1)+(2). Both Phase 2 and Phase 3 operate on this set.
    survivor_idx = np.where(keep)[0]
    pos_splat = arr[survivor_idx][:,
                   [name_to_idx["x"], name_to_idx["y"], name_to_idx["z"]]
               ].astype(np.float64)
    scales_splat = scales_splat_all[survivor_idx]                    # (M, 3)

    # ── Common: load dataparser + bounds, invert positions to metric ──
    need_metric = (not args.no_shape_filter) or (not args.no_utilization)
    pos_metric = None
    R_dp = t_dp = None
    dp_scale = colmap_to_metric = splat_to_metric = None
    if need_metric:
        dp_path = args.dataparser.resolve()
        if not dp_path.exists():
            print(f"ERROR: dataparser not found: {dp_path}", file=sys.stderr)
            return 2
        R_dp, t_dp, dp_scale = _load_dataparser_transform(dp_path)
        print(f"\n  dataparser: scale={dp_scale:.4f}, "
              f"R det={np.linalg.det(R_dp):.4f}")

        bp = args.bounds_json.resolve()
        if not bp.exists():
            print(f"ERROR: bounds-json not found: {bp}", file=sys.stderr)
            return 2
        bj = json.loads(bp.read_text())
        scale_factor = float(bj["scale_factor_da3_to_colmap"])
        colmap_to_metric = 1.0 / scale_factor
        splat_to_metric = colmap_to_metric / dp_scale
        print(f"  scale_factor_da3_to_colmap = {scale_factor:.6f}")
        print(f"  colmap_to_metric           = {colmap_to_metric:.6f}")
        print(f"  splat_to_metric            = {splat_to_metric:.6f}  "
              f"(1 splat unit = {splat_to_metric*100:.4f} cm)")

        pos_metric = _splat_to_metric(pos_splat, R_dp, t_dp, dp_scale,
                                      colmap_to_metric)
        print(f"  inverted {len(pos_metric):,} survivor positions to metric")
        print(f"    metric BB: "
              f"x[{pos_metric[:,0].min():.2f},{pos_metric[:,0].max():.2f}]  "
              f"y[{pos_metric[:,1].min():.2f},{pos_metric[:,1].max():.2f}]  "
              f"z[{pos_metric[:,2].min():.2f},{pos_metric[:,2].max():.2f}]")

    # ── Common: build raycasting scene (shared between v7 + v8) ───────
    scene = None
    face_normals_unit = None
    n_v = n_f = 0
    if need_metric:
        mp = args.scene_mesh.resolve()
        if not mp.exists():
            print(f"ERROR: scene mesh not found: {mp}", file=sys.stderr)
            return 2
        print(f"  loading scene mesh + building BVH (shared) ...")
        (scene, face_normals_unit, n_v, n_f,
         t_load, t_bvh) = _build_raycasting_scene_and_normals(mp)
        print(f"    mesh load: {t_load:.2f} s  ({n_v:,} verts, "
              f"{n_f:,} faces)")
        print(f"    BVH build: {t_bvh:.2f} s")

    # ── Filter 3: surface-normal V_perp (Phase 2 / v7) ────────────────
    mask_perp_survivors = None      # over survivor_idx
    if not args.no_shape_filter:
        rot_keys = ["rot_0", "rot_1", "rot_2", "rot_3"]   # (w, x, y, z)
        for k in rot_keys:
            if k not in name_to_idx:
                print(f"ERROR: PLY missing rotation property '{k}'",
                      file=sys.stderr)
                return 2
        quats = np.stack([arr[survivor_idx, name_to_idx[k]] for k in rot_keys],
                         axis=1)                                     # (M, 4)

        import open3d as o3d
        t0 = time.perf_counter()
        query = o3d.core.Tensor(pos_metric.astype(np.float32))
        cp_res = scene.compute_closest_points(query)
        prim_ids = cp_res["primitive_ids"].numpy().astype(np.int64)
        t_query = time.perf_counter() - t0
        print(f"\n  [Phase 2] closest-point query: {t_query:.2f} s "
              f"({len(prim_ids):,} points)")

        if prim_ids.min() < 0 or prim_ids.max() >= n_f:
            print(f"ERROR: primitive_ids out of range "
                  f"[{prim_ids.min()}, {prim_ids.max()}] vs n_faces={n_f}",
                  file=sys.stderr)
            return 2

        normals = face_normals_unit[prim_ids]                        # (M, 3)

        t0 = time.perf_counter()
        R_g = _quat_to_rot_matrix(quats)
        sigma_perp, sigma_max = _projected_perp_sigma(
            R_g, scales_splat, normals, splat_to_metric)
        t_cov = time.perf_counter() - t0
        print(f"  [Phase 2] covariance projection: {t_cov:.2f} s")

        sp_cm = sigma_perp * 100.0
        print(f"  [Phase 2] sigma_perp distribution (metric cm): "
              f"median {np.median(sp_cm):.3f}  "
              f"p90 {np.percentile(sp_cm, 90):.3f}  "
              f"p99 {np.percentile(sp_cm, 99):.3f}  "
              f"max {sp_cm.max():.3f}")

        mask_perp_survivors = sigma_perp <= args.max_perp_m
        n_keep_perp = int(mask_perp_survivors.sum())
        n_drop_perp = int((~mask_perp_survivors).sum())
        print(f"  [Phase 2] sigma_perp > {args.max_perp_m*100:.2f} cm: "
              f"drop {n_drop_perp:,} / keep {n_keep_perp:,} survivors "
              f"({100*n_drop_perp/max(n_pre_phase34,1):.1f}% dropped)")
    else:
        # Filter disabled: equivalent to "everyone passes V_perp".
        mask_perp_survivors = np.ones(n_pre_phase34, dtype=bool)

    # ── Filter 4: render-view utilization (Phase 3 / v8) ──────────────
    mask_util_survivors = None      # over survivor_idx
    if not args.no_utilization:
        sys.path.insert(0, str(Path(__file__).parent.resolve()))
        from scene_segmenter.views import V32ViewSource
        print(f"\n  [Phase 3] loading V32 view source from "
              f"{args.project_root} ...")
        t0 = time.perf_counter()
        vs = V32ViewSource(args.project_root,
                           bounds_json=args.bounds_json)
        views = vs.all_views()
        t_views = time.perf_counter() - t0
        print(f"    loaded {len(views)} views in {t_views:.2f} s")

        t0 = time.perf_counter()
        utilization = _count_views_per_point(scene, pos_metric, views,
                                              args.depth_tol_m)
        t_util = time.perf_counter() - t0
        print(f"\n  [Phase 3] utilization computed in {t_util:.2f} s "
              f"({len(pos_metric):,} pts x {len(views)} views)")

        n_views = len(views)
        med = float(np.median(utilization))
        p10 = float(np.percentile(utilization, 10))
        p25 = float(np.percentile(utilization, 25))
        p50 = float(np.percentile(utilization, 50))
        p90 = float(np.percentile(utilization, 90))
        umax = int(utilization.max())
        umin = int(utilization.min())
        print(f"  [Phase 3] utilization distribution (out of {n_views} "
              f"views):")
        print(f"    min  = {umin}   p10 = {p10:.0f}   p25 = {p25:.0f}   "
              f"median = {p50:.0f}   p90 = {p90:.0f}   max = {umax}")
        n_zero = int((utilization == 0).sum())
        n_lt_min = int((utilization < args.min_views).sum())
        print(f"    n with 0 views: {n_zero:,} "
              f"({100*n_zero/max(len(utilization),1):.1f}%)")
        print(f"    n with < {args.min_views} views: {n_lt_min:,} "
              f"({100*n_lt_min/max(len(utilization),1):.1f}%)")

        mask_util_survivors = utilization >= args.min_views
        n_keep_util = int(mask_util_survivors.sum())
        n_drop_util = int((~mask_util_survivors).sum())
        print(f"  [Phase 3] utilization < {args.min_views} views: "
              f"drop {n_drop_util:,} / keep {n_keep_util:,} survivors "
              f"({100*n_drop_util/max(n_pre_phase34,1):.1f}% dropped)")
    else:
        # Filter disabled: equivalent to "everyone passes utilization".
        mask_util_survivors = np.ones(n_pre_phase34, dtype=bool)

    # ── Combine masks (intersection or union) ─────────────────────────
    if args.mode == "intersection":
        mask_combined_survivors = mask_perp_survivors & mask_util_survivors
        combine_op = "AND"
    else:
        mask_combined_survivors = mask_perp_survivors | mask_util_survivors
        combine_op = "OR"

    # Cross-tab of perp vs util on the survivor set (only meaningful if
    # both filters were actually evaluated).
    if (not args.no_shape_filter) and (not args.no_utilization):
        a = mask_perp_survivors
        b = mask_util_survivors
        nn_pass_pass = int(( a &  b).sum())
        nn_pass_fail = int(( a & ~b).sum())   # on-surface shape but invisible
        nn_fail_pass = int((~a &  b).sum())   # mis-oriented but visible
        nn_fail_fail = int((~a & ~b).sum())   # both fail (=most aggressive drop)
        print(f"\n  cross-tab on {n_pre_phase34:,} (1+2)-survivors "
              f"(rows=V_perp, cols=utilization):")
        print(f"    {'':12s}  util_pass  util_fail")
        print(f"    perp_pass   {nn_pass_pass:>9,}  {nn_pass_fail:>9,}"
              f"   <- on-surface shape, "
              f"{'visible' if combine_op=='AND' else 'pass union'}")
        print(f"    perp_fail   {nn_fail_pass:>9,}  {nn_fail_fail:>9,}"
              f"   <- mis-oriented; both-fail = strongest drop signal")
        print(f"    intersection (AND) keep: {nn_pass_pass:,}")
        print(f"    union (OR) keep:         "
              f"{nn_pass_pass + nn_pass_fail + nn_fail_pass:,}")

    n_keep_combined = int(mask_combined_survivors.sum())
    n_drop_combined = int((~mask_combined_survivors).sum())
    pct_drop_combined = 100.0 * n_drop_combined / max(n_pre_phase34, 1)
    print(f"\n  combined (mode={args.mode}, op={combine_op}): "
          f"keep {n_keep_combined:,} / drop {n_drop_combined:,} "
          f"survivors ({pct_drop_combined:.1f}% dropped)")

    # Project survivor-mask back onto full keep array.
    full_combined_mask = np.zeros(n, dtype=bool)
    full_combined_mask[survivor_idx] = mask_combined_survivors
    keep &= full_combined_mask

    # ── Per-filter summary ────────────────────────────────────────────
    n_drop_perp_only = (int((~mask_perp_survivors).sum())
                        if not args.no_shape_filter else 0)
    n_drop_util_only = (int((~mask_util_survivors).sum())
                        if not args.no_utilization else 0)
    print(f"\n  per-filter drop counts:")
    print(f"    filter 1 (scale > p{args.scale_pct}): {dropped_scale:,}")
    print(f"    filter 2 (opacity < {args.opacity_min}): {dropped_op:,}")
    print(f"    filter 3 (V_perp > {args.max_perp_m*100:.2f} cm) "
          f"on (1+2)-survivors: {n_drop_perp_only:,}")
    print(f"    filter 4 (util < {args.min_views} views) "
          f"on (1+2)-survivors: {n_drop_util_only:,}")
    print(f"    combined (mode={args.mode}): {n_drop_combined:,}")

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
