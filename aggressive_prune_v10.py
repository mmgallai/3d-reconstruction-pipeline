"""
aggressive_prune_v10 — soft AABB falloff via opacity modulation.

DESIGN (per PI recommendation)
==============================
Instead of *dropping* Gaussians whose centres fall outside the scene-mesh
AABB (v3's behaviour), we *modulate* their opacity via a smoothstep
falloff. The ellipses stay whole and physically present, but their
outer edges are driven toward alpha=0, so they seamlessly blend into
the background instead of leaving hard-edged beams / silhouettes at
the crop boundary.

BOUNDARY GEOMETRY
=================
  inner AABB  = scene-mesh AABB (splat-space) expanded by `margin_m` on
                every face. Inside this box: opacity multiplier = 1.
  outer AABB  = inner AABB expanded by `falloff_m` on every face.
                At outer boundary: opacity multiplier = 0.
  transition  = smoothstep on the Euclidean distance from the Gaussian
                centre to the inner AABB surface (0 inside, positive
                outside), normalised by `falloff_edge_splat`.

PER-GAUSSIAN MATH
=================
    dx = max(inner_x_min - x, 0, x - inner_x_max)
    dy = max(inner_y_min - y, 0, y - inner_y_max)
    dz = max(inner_z_min - z, 0, z - inner_z_max)
    dist_outside = sqrt(dx^2 + dy^2 + dz^2)          # 0 if inside inner AABB

    t     = clip(dist_outside / falloff_edge_splat, 0, 1)
    mult  = 1 - t*t*(3 - 2*t)                        # smoothstep, 1 -> 0

    sig       = sigmoid(opacity_logit)
    new_sig   = sig * mult
    new_logit = log(new_sig / (1 - new_sig))         # numerically clamped

USAGE
=====
    python aggressive_prune_v10.py [input_ply] \\
        [--out output/splat_v32_data3_v10.ply] \\
        [--scene-mesh output/mesh_v32_data3/mesh_v32_data3_openmvs.ply] \\
        [--dataparser nerfstudio/dense/splatfacto/2026-06-07_203807/dataparser_transforms.json] \\
        [--bounds-json colmap/dense/tof_bounds.json] \\
        [--margin-m 0.10] [--falloff-m 0.10] [--drop-if-below 0.01]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Reuse the v1 PLY reader/writer.
sys.path.insert(0, str(Path(__file__).parent))
from aggressive_prune import _read_ply, _write_ply                    # noqa: E402
# Reuse the metric<->splat transform helpers.
from _spatial_crop_splat import _load_dataparser_transform            # noqa: E402


# ---------------------------------------------------------------------------
# Metric -> splat forward transform (same convention as aggressive_prune_v3).
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
                   help="Output .ply. Default: <input>_v10.ply")
    p.add_argument("--scene-mesh", type=Path,
                   default=Path("output/mesh_v32_data3/mesh_v32_data3_openmvs.ply"),
                   help="OpenMVS scene mesh whose vertex AABB defines the "
                        "inner boundary (default V32 data3 mesh)")
    p.add_argument("--dataparser", type=Path,
                   default=Path("nerfstudio/dense/splatfacto/2026-06-07_203807/dataparser_transforms.json"),
                   help="nerfstudio dataparser_transforms.json for the splat "
                        "run (default V32 data3 run)")
    p.add_argument("--bounds-json", type=Path,
                   default=Path("colmap/dense/tof_bounds.json"),
                   help="tof_bounds.json containing scale_factor_da3_to_colmap")
    p.add_argument("--margin-m", type=float, default=0.10,
                   help="Metric margin (m) added to every face of the mesh "
                        "AABB. Inside this expanded box, opacity mult = 1. "
                        "Default 0.10 m.")
    p.add_argument("--falloff-m", type=float, default=0.10,
                   help="Metric width (m) of the smoothstep falloff. "
                        "Opacity mult goes from 1 at margin edge to 0 at "
                        "margin + falloff. Default 0.10 m.")
    p.add_argument("--drop-if-below", type=float, default=0.01,
                   help="Numeric safety: drop Gaussians whose new "
                        "sigmoid(opacity) falls below this threshold. Keeps "
                        "file size sane. Set to 0 to keep all. Default 0.01.")
    p.add_argument("--aabb-percentile", type=float, default=1.0,
                   help="Percentile clip on scene-mesh vertices (per axis) "
                        "before taking the AABB. Same convention as "
                        "aggressive_prune_v3. Default 1.0.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Scene-mesh AABB in splat space (same routine as aggressive_prune_v3, but
# returns unmargined bounds and lets the caller apply margin explicitly).
# ---------------------------------------------------------------------------
def _mesh_aabb_in_splat(scene_mesh_path: Path, R, t, dp_scale,
                        colmap_to_metric: float, percentile: float):
    import trimesh
    print(f"  scene mesh: {scene_mesh_path}")
    m = trimesh.load(str(scene_mesh_path), force="mesh", process=False)
    v_metric = np.asarray(m.vertices, dtype=np.float64)
    print(f"    {len(v_metric):,} vertices in metric world")
    print(f"    metric range: "
          f"x[{v_metric[:,0].min():+.3f},{v_metric[:,0].max():+.3f}]  "
          f"y[{v_metric[:,1].min():+.3f},{v_metric[:,1].max():+.3f}]  "
          f"z[{v_metric[:,2].min():+.3f},{v_metric[:,2].max():+.3f}]")
    v_splat = _metric_to_splat(v_metric, R, t, dp_scale, colmap_to_metric)
    lo = np.percentile(v_splat, percentile, axis=0)
    hi = np.percentile(v_splat, 100.0 - percentile, axis=0)
    return lo.astype(np.float64), hi.astype(np.float64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = _parse_args()
    in_path = args.input_ply.resolve()
    out_path = (args.out or in_path.with_name(in_path.stem + "_v10.ply")).resolve()
    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        return 2

    print(f"=== aggressive_prune_v10 (soft AABB falloff) ===")
    print(f"  input  : {in_path}")
    print(f"  output : {out_path}")

    # ---- Load splat ---------------------------------------------------------
    n, props, raw, rec_size, header = _read_ply(in_path)
    arr = np.frombuffer(raw, dtype=np.float32).reshape(n, len(props)).copy()
    name_to_idx = {p: i for i, p in enumerate(props)}
    print(f"  loaded {n:,} Gaussians, {len(props)} properties each")

    x = arr[:, name_to_idx["x"]].astype(np.float64)
    y = arr[:, name_to_idx["y"]].astype(np.float64)
    z = arr[:, name_to_idx["z"]].astype(np.float64)

    # ---- Load dataparser + bounds ------------------------------------------
    R, t, dp_scale = _load_dataparser_transform(args.dataparser)
    bounds = json.loads(args.bounds_json.read_text())
    colmap_to_metric = 1.0 / float(bounds["scale_factor_da3_to_colmap"])
    metric_to_splat = dp_scale / colmap_to_metric
    print(f"  dp_scale={dp_scale:.6f}  colmap_to_metric={colmap_to_metric:.6f}  "
          f"metric_to_splat={metric_to_splat:.6f}")

    # ---- Build mesh AABB (before margin) -----------------------------------
    lo0, hi0 = _mesh_aabb_in_splat(args.scene_mesh, R, t, dp_scale,
                                   colmap_to_metric, args.aabb_percentile)
    print(f"  splat-space AABB (mesh, {args.aabb_percentile:.1f}-"
          f"{100.0-args.aabb_percentile:.1f} percentile): "
          f"x[{lo0[0]:+.3f},{hi0[0]:+.3f}]  "
          f"y[{lo0[1]:+.3f},{hi0[1]:+.3f}]  "
          f"z[{lo0[2]:+.3f},{hi0[2]:+.3f}]")

    # ---- Expand by margin (inner AABB) -------------------------------------
    margin_splat = args.margin_m * metric_to_splat
    lo = lo0 - margin_splat
    hi = hi0 + margin_splat
    print(f"  inner AABB (+{args.margin_m*100:.1f} cm = "
          f"{margin_splat:.4f} splat units margin): "
          f"x[{lo[0]:+.3f},{hi[0]:+.3f}]  "
          f"y[{lo[1]:+.3f},{hi[1]:+.3f}]  "
          f"z[{lo[2]:+.3f},{hi[2]:+.3f}]")

    # ---- Outer falloff distance -------------------------------------------
    falloff_edge_splat = args.falloff_m * metric_to_splat
    print(f"  falloff edge = {args.falloff_m*100:.1f} cm "
          f"({falloff_edge_splat:.4f} splat units)")
    # Outer AABB (informational): distance from inner surface at which mult=0.
    outer_lo = lo - falloff_edge_splat
    outer_hi = hi + falloff_edge_splat
    print(f"  outer AABB (opacity mult = 0 boundary): "
          f"x[{outer_lo[0]:+.3f},{outer_hi[0]:+.3f}]  "
          f"y[{outer_lo[1]:+.3f},{outer_hi[1]:+.3f}]  "
          f"z[{outer_lo[2]:+.3f},{outer_hi[2]:+.3f}]")

    # ---- Per-Gaussian distance-outside inner AABB --------------------------
    dx = np.maximum.reduce([lo[0] - x, np.zeros_like(x), x - hi[0]])
    dy = np.maximum.reduce([lo[1] - y, np.zeros_like(y), y - hi[1]])
    dz = np.maximum.reduce([lo[2] - z, np.zeros_like(z), z - hi[2]])
    dist_outside = np.sqrt(dx * dx + dy * dy + dz * dz)                # (N,)

    outside_mask = dist_outside > 0
    n_outside = int(outside_mask.sum())
    n_inside = n - n_outside
    print(f"\n  Gaussian centres: {n_inside:,} inside inner AABB "
          f"({100.0*n_inside/n:.1f}%), {n_outside:,} outside "
          f"({100.0*n_outside/n:.1f}%)")

    if n_outside > 0:
        out_vals = dist_outside[outside_mask]
        med = float(np.median(out_vals))
        p90 = float(np.percentile(out_vals, 90))
        mx = float(out_vals.max())
        print(f"  dist_outside (splat units, outside centres only): "
              f"median {med:.4f}  p90 {p90:.4f}  max {mx:.4f}")
        splat_to_cm = (colmap_to_metric / dp_scale) * 100.0
        print(f"  dist_outside (metric cm equivalent): "
              f"median {med*splat_to_cm:.2f}  "
              f"p90 {p90*splat_to_cm:.2f}  "
              f"max {mx*splat_to_cm:.2f}")

    # ---- Smoothstep multiplier --------------------------------------------
    if falloff_edge_splat <= 0:
        # Falloff disabled: multiplier is a hard 1 inside / 0 outside.
        mult = np.where(dist_outside > 0, 0.0, 1.0)
    else:
        t_ss = np.clip(dist_outside / falloff_edge_splat, 0.0, 1.0)
        mult = 1.0 - t_ss * t_ss * (3.0 - 2.0 * t_ss)                  # smoothstep
    mult = mult.astype(np.float64)

    # ---- Multiplier histogram ---------------------------------------------
    n_full = int((mult >= 0.99).sum())
    n_partial = int(((mult >= 0.01) & (mult < 0.99)).sum())
    n_zeroed = int((mult < 0.01).sum())
    print(f"\n  opacity multiplier distribution:")
    print(f"    full     (mult >= 0.99): {n_full:,}  ({100.0*n_full/n:.1f}%)")
    print(f"    partial  (0.01..0.99):   {n_partial:,}  ({100.0*n_partial/n:.1f}%)")
    print(f"    zeroed   (mult <  0.01): {n_zeroed:,}  ({100.0*n_zeroed/n:.1f}%)")

    # ---- Apply multiplier to opacity logit --------------------------------
    op_idx = name_to_idx["opacity"]
    op_logit = arr[:, op_idx].astype(np.float64)
    sig = 1.0 / (1.0 + np.exp(-op_logit))
    new_sig = sig * mult
    # Numerical safety: clamp new_sig strictly inside (0, 1) before logit.
    eps = 1e-7
    new_sig_clamped = np.clip(new_sig, eps, 1.0 - eps)
    new_logit = np.log(new_sig_clamped / (1.0 - new_sig_clamped))

    print(f"\n  opacity BEFORE: min={sig.min():.4f}  median={np.median(sig):.4f}  "
          f"max={sig.max():.4f}")
    print(f"  opacity AFTER : min={new_sig.min():.4f}  median={np.median(new_sig):.4f}  "
          f"max={new_sig.max():.4f}")

    # ---- Drop-if-below safety net -----------------------------------------
    if args.drop_if_below > 0:
        keep_mask = new_sig >= args.drop_if_below
        n_dropped = int((~keep_mask).sum())
        print(f"\n  drop-if-below (sigmoid < {args.drop_if_below}): "
              f"drop {n_dropped:,} Gaussians "
              f"({100.0*n_dropped/n:.2f}%)")
    else:
        keep_mask = np.ones(n, dtype=bool)
        print(f"\n  drop-if-below disabled -- keeping all {n:,} Gaussians")

    # ---- Write ------------------------------------------------------------
    arr[:, op_idx] = new_logit.astype(np.float32)
    kept = arr[keep_mask]
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
