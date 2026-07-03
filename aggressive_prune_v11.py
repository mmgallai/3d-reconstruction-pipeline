"""
aggressive_prune_v11 — mesh-distance SDF-like soft SHRINK of scale (and
optionally opacity).

DESIGN (per PI recommendation, adapted for open OpenMVS meshes)
===============================================================
PI's original ask was to build a signed distance field (SDF) over the
scene mesh and shrink / kill any Gaussian whose SDF value exceeds zero
(i.e. lies outside the mesh volume). Our OpenMVS mesh is an OPEN surface
(not watertight), so a true signed SDF is undefined -- there is no
meaningful "inside" volume to sign against. We approximate the intent by
using UNSIGNED distance to the nearest mesh face
(``open3d.t.geometry.RaycastingScene.compute_distance``), the same
primitive v5 uses. This is well-defined for any triangulated mesh.

The distinguishing levers vs the existing pruners:

  * v5:  HARD threshold drop on mesh distance (>3 cm -> gone).
  * v10: soft AABB opacity falloff (mesh AABB, not surface).
  * v11: SMOOTH shrink of the per-Gaussian SCALE (and optionally opacity)
         as mesh distance grows from an inner edge to an outer edge.

Because gsplat stores log(scale), the multiplicative shrink is simply an
additive term on the log-scale channels:

    scale_log_new = scale_log + log(mult)      # elementwise
    equivalent to: scale_new = scale * mult

The falloff is a smoothstep on distance:

    inner_edge_splat = inner_m * metric_to_splat
    outer_edge_splat = outer_m * metric_to_splat
    t                = clip((dist - inner_edge_splat) /
                            (outer_edge_splat - inner_edge_splat), 0, 1)
    mult             = 1 - t*t*(3 - 2*t)       # smoothstep: 1 -> 0

    mode 'scale'   :  shrinks scale_0/1/2 log by log(mult)
    mode 'opacity' :  shrinks sigmoid(opacity) by mult (same as v10)
    mode 'both'    :  applies both shrinks

TRANSFORM CHOICE
================
v5 inverse-transforms every Gaussian back to metric world to query the
mesh. That works but round-trips through the dataparser twice. v10
forward-transforms the MESH into splat space once and queries directly
in splat units. v11 follows v10's approach for simplicity + precision:
mesh vertices go metric -> colmap -> splat once at load time, and all
distance queries happen in splat units. Distances are converted to cm
only for human-readable diagnostics.

CLI
===
    python aggressive_prune_v11.py [input_ply] \\
        [--out output/splat_v32_data3_v11.ply] \\
        [--scene-mesh output/mesh_v32_data3/mesh_v32_data3_openmvs.ply] \\
        [--dataparser nerfstudio/dense/splatfacto/2026-06-07_203807/dataparser_transforms.json] \\
        [--bounds-json colmap/dense/tof_bounds.json] \\
        [--mode {scale,opacity,both}]  default 'scale' \\
        [--inner-m 0.02] [--outer-m 0.20] \\
        [--drop-if-below 0.01] [--scale-drop-mm 0.5]

Do NOT touch the input file. A fresh .ply is written.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Reuse the v1 PLY reader/writer.
sys.path.insert(0, str(Path(__file__).parent))
from aggressive_prune import _read_ply, _write_ply                    # noqa: E402
# Reuse the dataparser helper (same convention v3/v5/v10 use).
from _spatial_crop_splat import _load_dataparser_transform            # noqa: E402


# ---------------------------------------------------------------------------
# Metric -> splat forward transform (same convention as v3 / v10).
# ---------------------------------------------------------------------------
def _metric_to_splat(positions_metric: np.ndarray,
                     R: np.ndarray, t: np.ndarray,
                     dp_scale: float, colmap_to_metric: float) -> np.ndarray:
    """Forward transform metric -> splat space.

        colmap_pos = metric_pos / colmap_to_metric
        splat_pos  = dp_scale * (R @ colmap_pos + t)
    """
    colmap = positions_metric / colmap_to_metric
    splat = dp_scale * (colmap @ R.T + t[None, :])
    return splat.astype(np.float64)


# ---------------------------------------------------------------------------
# Load mesh, forward-transform into splat space, wrap in a RaycastingScene.
# ---------------------------------------------------------------------------
def _build_raycasting_scene_in_splat(mesh_path: Path,
                                     R: np.ndarray, t: np.ndarray,
                                     dp_scale: float,
                                     colmap_to_metric: float):
    """Load a triangulated mesh, forward-transform its vertices into
    splat space, and wrap it in an Open3D RaycastingScene BVH. Returns
    (scene, n_verts, n_faces, t_load_s, t_bvh_s)."""
    import open3d as o3d
    import trimesh

    t0 = time.perf_counter()
    m = trimesh.load(str(mesh_path), force="mesh", process=False)
    if m.is_empty or len(m.faces) == 0:
        raise RuntimeError(f"Scene mesh has no faces: {mesh_path}")
    v_metric = np.asarray(m.vertices, dtype=np.float64)
    faces = np.asarray(m.faces, dtype=np.uint32)
    v_splat = _metric_to_splat(v_metric, R, t, dp_scale, colmap_to_metric)
    t_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    tmesh = o3d.t.geometry.TriangleMesh()
    tmesh.vertex.positions = o3d.core.Tensor(v_splat.astype(np.float32))
    tmesh.triangle.indices = o3d.core.Tensor(faces)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)
    t_bvh = time.perf_counter() - t0

    return scene, len(v_splat), len(faces), t_load, t_bvh, v_splat


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "input_ply", type=Path, nargs="?",
        default=Path("output/splat_v32_data3_noinit_pruned.ply"),
        help="Input splatfacto .ply. Default: "
             "output/splat_v32_data3_noinit_pruned.ply")
    p.add_argument("--out", type=Path, default=None,
                   help="Output .ply. Default: <input>_v11.ply")
    p.add_argument("--scene-mesh", type=Path,
                   default=Path("output/mesh_v32_data3/"
                                "mesh_v32_data3_openmvs.ply"),
                   help="OpenMVS scene mesh whose faces define the "
                        "reference surface (default V32 data3 mesh)")
    p.add_argument("--dataparser", type=Path,
                   default=Path("nerfstudio/dense/splatfacto/"
                                "2026-06-07_203807/"
                                "dataparser_transforms.json"),
                   help="nerfstudio dataparser_transforms.json for the "
                        "splat run (default V32 data3 run)")
    p.add_argument("--bounds-json", type=Path,
                   default=Path("colmap/dense/tof_bounds.json"),
                   help="tof_bounds.json containing "
                        "scale_factor_da3_to_colmap")
    p.add_argument("--mode", choices=("scale", "opacity", "both"),
                   default="scale",
                   help="Which Gaussian parameter to shrink as distance "
                        "grows. Default 'scale'.")
    p.add_argument("--inner-m", type=float, default=0.02,
                   help="Metric distance (m) below which the multiplier "
                        "is exactly 1 (Gaussians untouched). Default "
                        "0.02 m = 2 cm.")
    p.add_argument("--outer-m", type=float, default=0.20,
                   help="Metric distance (m) at which the multiplier "
                        "reaches 0. Default 0.20 m = 20 cm.")
    p.add_argument("--drop-if-below", type=float, default=0.01,
                   help="Drop Gaussians whose sigmoid(opacity) falls "
                        "below this threshold after the shrink. Set 0 "
                        "to disable. Default 0.01.")
    p.add_argument("--scale-drop-mm", type=float, default=0.5,
                   help="In 'scale'/'both' mode, additionally drop "
                        "Gaussians whose max metric sigma drops below "
                        "this many mm. Set 0 to disable. Default 0.5 mm.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = _parse_args()
    in_path = args.input_ply.resolve()
    out_path = (args.out or
                in_path.with_name(in_path.stem + "_v11.ply")).resolve()
    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        return 2
    if args.outer_m <= args.inner_m:
        print(f"ERROR: --outer-m ({args.outer_m}) must be greater than "
              f"--inner-m ({args.inner_m})", file=sys.stderr)
        return 2

    print(f"=== aggressive_prune_v11 (mesh-distance soft shrink) ===")
    print(f"  input   : {in_path}")
    print(f"  output  : {out_path}")
    print(f"  mode    : {args.mode}")

    # ---- Load splat ---------------------------------------------------------
    n, props, raw, rec_size, header = _read_ply(in_path)
    arr = np.frombuffer(raw, dtype=np.float32).reshape(n, len(props)).copy()
    name_to_idx = {p: i for i, p in enumerate(props)}
    print(f"  loaded {n:,} Gaussians, {len(props)} properties each")

    x = arr[:, name_to_idx["x"]].astype(np.float64)
    y = arr[:, name_to_idx["y"]].astype(np.float64)
    z = arr[:, name_to_idx["z"]].astype(np.float64)
    centres_splat = np.stack([x, y, z], axis=1)                       # (N, 3)

    op_idx = name_to_idx["opacity"]
    op_logit = arr[:, op_idx].astype(np.float64)
    sig = 1.0 / (1.0 + np.exp(-op_logit))

    scale_keys = sorted([p for p in props if p.startswith("scale_")])
    scale_indices = [name_to_idx[k] for k in scale_keys]
    scale_log = arr[:, scale_indices].astype(np.float64)              # (N, 3)

    # ---- Load dataparser + bounds ------------------------------------------
    R, t, dp_scale = _load_dataparser_transform(args.dataparser)
    bounds = json.loads(args.bounds_json.read_text())
    colmap_to_metric = 1.0 / float(bounds["scale_factor_da3_to_colmap"])
    metric_to_splat = dp_scale / colmap_to_metric
    splat_to_metric = 1.0 / metric_to_splat
    print(f"  dp_scale={dp_scale:.6f}  colmap_to_metric={colmap_to_metric:.6f}"
          f"  metric_to_splat={metric_to_splat:.6f}")

    # ---- Splat-space AABB (for sanity) -------------------------------------
    lo_g = centres_splat.min(axis=0)
    hi_g = centres_splat.max(axis=0)
    print(f"  splat-space Gaussian AABB: "
          f"x[{lo_g[0]:+.3f},{hi_g[0]:+.3f}]  "
          f"y[{lo_g[1]:+.3f},{hi_g[1]:+.3f}]  "
          f"z[{lo_g[2]:+.3f},{hi_g[2]:+.3f}]")

    # ---- Build RaycastingScene over mesh (in splat space) ------------------
    print(f"  scene mesh: {args.scene_mesh}")
    scene, n_v, n_f, t_load, t_bvh, v_splat = _build_raycasting_scene_in_splat(
        args.scene_mesh, R, t, dp_scale, colmap_to_metric)
    print(f"    mesh load + forward-transform: {t_load:.2f} s  "
          f"({n_v:,} verts, {n_f:,} faces)")
    print(f"    BVH build:                     {t_bvh:.2f} s")
    print(f"    mesh splat-space AABB: "
          f"x[{v_splat[:,0].min():+.3f},{v_splat[:,0].max():+.3f}]  "
          f"y[{v_splat[:,1].min():+.3f},{v_splat[:,1].max():+.3f}]  "
          f"z[{v_splat[:,2].min():+.3f},{v_splat[:,2].max():+.3f}]")

    # ---- Compute inner / outer edges ---------------------------------------
    inner_edge_splat = args.inner_m * metric_to_splat
    outer_edge_splat = args.outer_m * metric_to_splat
    print(f"\n  falloff window (metric):  inner {args.inner_m*100:.1f} cm  "
          f"outer {args.outer_m*100:.1f} cm")
    print(f"  falloff window (splat):   inner {inner_edge_splat:.4f}  "
          f"outer {outer_edge_splat:.4f}")

    # ---- Distance query ----------------------------------------------------
    import open3d as o3d
    t0 = time.perf_counter()
    query = o3d.core.Tensor(centres_splat.astype(np.float32))
    dist_splat = scene.compute_distance(query).numpy().astype(np.float64)
    t_query = time.perf_counter() - t0
    print(f"  distance query: {t_query:.2f} s ({n:,} points)")

    # ---- Distance histogram -----------------------------------------------
    d_cm = dist_splat * splat_to_metric * 100.0
    def _pct(a, p):
        return float(np.percentile(a, p))
    d_med_s = float(np.median(dist_splat))
    print(f"\n  distance to nearest face:")
    print(f"    splat units: p10 {_pct(dist_splat,10):.4f}  "
          f"p50 {d_med_s:.4f}  p90 {_pct(dist_splat,90):.4f}  "
          f"p99 {_pct(dist_splat,99):.4f}  max {dist_splat.max():.4f}")
    print(f"    metric cm  : p10 {_pct(d_cm,10):.2f}  "
          f"p50 {float(np.median(d_cm)):.2f}  p90 {_pct(d_cm,90):.2f}  "
          f"p99 {_pct(d_cm,99):.2f}  max {d_cm.max():.2f}")

    # ---- Smoothstep multiplier --------------------------------------------
    denom = outer_edge_splat - inner_edge_splat
    t_ss = np.clip((dist_splat - inner_edge_splat) / denom, 0.0, 1.0)
    mult = 1.0 - t_ss * t_ss * (3.0 - 2.0 * t_ss)                     # (N,)

    # Multiplier histogram.
    n_full = int((mult >= 0.99).sum())
    n_partial = int(((mult >= 0.01) & (mult < 0.99)).sum())
    n_zeroed = int((mult < 0.01).sum())
    print(f"\n  multiplier distribution:")
    print(f"    full     (mult >= 0.99): {n_full:,}  "
          f"({100.0*n_full/n:.1f}%)")
    print(f"    partial  (0.01..0.99):   {n_partial:,}  "
          f"({100.0*n_partial/n:.1f}%)")
    print(f"    zeroed   (mult <  0.01): {n_zeroed:,}  "
          f"({100.0*n_zeroed/n:.1f}%)")

    # ---- Apply multiplier(s) ----------------------------------------------
    # Clamp mult away from exactly zero to avoid log(0) / logit(0).
    mult_clamped = np.clip(mult, 1e-6, 1.0)
    log_mult = np.log(mult_clamped)                                   # (N,)

    # Snapshot BEFORE for reporting.
    sigma_max_before_mm = (np.exp(scale_log).max(axis=1)
                           * splat_to_metric * 1000.0)
    sig_mean_before = float(sig.mean())
    print(f"\n  BEFORE:")
    print(f"    scale log mean (all channels):  {scale_log.mean():.4f}")
    print(f"    sigma_max metric mm: min {sigma_max_before_mm.min():.3f}  "
          f"median {float(np.median(sigma_max_before_mm)):.3f}  "
          f"max {sigma_max_before_mm.max():.3f}")
    print(f"    opacity sigmoid mean:           {sig_mean_before:.4f}")

    new_scale_log = scale_log.copy()
    new_sig = sig.copy()
    new_op_logit = op_logit.copy()

    if args.mode in ("scale", "both"):
        # Shrink all three scale channels by the same multiplicative factor.
        new_scale_log = scale_log + log_mult[:, None]
    if args.mode in ("opacity", "both"):
        new_sig = sig * mult
        eps = 1e-7
        new_sig_clamped = np.clip(new_sig, eps, 1.0 - eps)
        new_op_logit = np.log(new_sig_clamped / (1.0 - new_sig_clamped))

    # AFTER stats.
    sigma_max_after_mm = (np.exp(new_scale_log).max(axis=1)
                          * splat_to_metric * 1000.0)
    print(f"  AFTER :")
    print(f"    scale log mean (all channels):  {new_scale_log.mean():.4f}")
    print(f"    sigma_max metric mm: min {sigma_max_after_mm.min():.4f}  "
          f"median {float(np.median(sigma_max_after_mm)):.3f}  "
          f"max {sigma_max_after_mm.max():.3f}")
    print(f"    opacity sigmoid mean:           {float(new_sig.mean()):.4f}")

    # ---- Drop safety nets --------------------------------------------------
    keep = np.ones(n, dtype=bool)

    # Treat any mult that hit the clamp floor (was ~0) as a delete flag,
    # regardless of mode -- the smoothstep has driven it to zero, so the
    # Gaussian is effectively gone.
    zeroed = mult < 1e-6
    n_zero_flag = int(zeroed.sum())
    if n_zero_flag:
        print(f"\n  hard-delete flag (mult == 0 exactly): "
              f"drop {n_zero_flag:,} Gaussians")
        keep &= ~zeroed

    if args.drop_if_below > 0:
        op_drop = new_sig < args.drop_if_below
        n_op_drop = int((op_drop & keep).sum())
        print(f"  drop-if-below (sigmoid < {args.drop_if_below}): "
              f"drop {n_op_drop:,} additional Gaussians")
        keep &= ~op_drop
    else:
        print(f"  drop-if-below disabled")

    if args.mode in ("scale", "both") and args.scale_drop_mm > 0:
        scale_drop = sigma_max_after_mm < args.scale_drop_mm
        n_scale_drop = int((scale_drop & keep).sum())
        print(f"  scale-drop (sigma_max < {args.scale_drop_mm} mm): "
              f"drop {n_scale_drop:,} additional Gaussians")
        keep &= ~scale_drop
    elif args.mode in ("scale", "both"):
        print(f"  scale-drop disabled")

    # ---- Write shrunk parameters back into arr -----------------------------
    for i, col in enumerate(scale_indices):
        arr[:, col] = new_scale_log[:, i].astype(np.float32)
    arr[:, op_idx] = new_op_logit.astype(np.float32)

    kept = arr[keep]
    n_kept = kept.shape[0]
    print(f"\n  final: n_kept={n_kept:,} / n_input={n:,} "
          f"({100.0*n_kept/n:.2f}%)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_ply(out_path, header, kept)
    size_mb = out_path.stat().st_size / 1_048_576
    print(f"  saved: {out_path}  ({size_mb:.1f} MB)")
    print(f"\n=== Done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
