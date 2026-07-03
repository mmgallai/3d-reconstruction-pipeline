"""
aggressive_prune_v3 — peripheral-floater cleanup using a SCENE-MESH-derived
AABB (instead of v2's object-only AABB).

WHAT CHANGED vs v2
==================

v2 derived the AABB from the union of per-object splats (bottle + box +
lobster) plus a 30 cm metric margin. On V32 this turned out to be too
tight: the back wall and the desk's far perimeter were clipped because
no object sits against them. The objects' AABB is ~80 × 35 × 45 cm —
much smaller than the full reconstructed scene.

v3 derives the AABB from the V32 SCENE MESH vertices instead. Why this is
the right source:

  • The scene mesh (`output/mesh_<scene>/mesh_<scene>_openmvs.ply`) is the
    faithful 3D geometry of the desk + back wall + every surface OpenMVS
    could actually reconstruct.
  • Photometric floaters (leafy specular fringe, monitor reflection blobs)
    do NOT exist in the mesh because they have no real 3D geometry —
    OpenMVS can't triangulate them. So the mesh's spatial extent IS
    exactly the "everything real" envelope, by construction.
  • A small percentile clip (1st–99th, by default) drops stray mesh
    outliers (a few erratic faces in the OpenMVS output).

Algorithm:
  1. Read scene mesh vertices in metric world coordinates.
  2. Apply the same forward transform the splat used (metric → COLMAP units →
     splat-space via `dataparser_transforms.json`).
  3. Take a 1-99 percentile AABB on each axis.
  4. Expand by a small metric margin (default 10 cm — we already have a
     scene-tight starting box).
  5. Run the same scale + opacity prune as v1/v2 (defaults unchanged).
  6. Clip Gaussians outside the AABB.

USAGE
=====
    python aggressive_prune_v3.py output/splat_v32_data3_noinit_pruned.ply \
        [--out output/splat_v32_data3_v3.ply]

Default `--scene-mesh`, `--dataparser`, `--bounds-json` paths target V32
data3. Override for other scenes.

OPTIONAL OVERRIDES
==================
  --scene-mesh <path>     path to V32 OpenMVS mesh
  --dataparser <path>     dataparser_transforms.json from the splatfacto run
  --bounds-json <path>    tof_bounds.json (for scale_factor_da3_to_colmap)
  --aabb-percentile 1.0   bottom percentile to clip mesh outliers (and 100 - this)
  --margin-m 0.10         metric margin to expand the percentile AABB
  --no-aabb               disable AABB step entirely
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Reuse v1's PLY reader/writer.
sys.path.insert(0, str(Path(__file__).parent))
from aggressive_prune import _read_ply, _write_ply                  # noqa: E402
# Reuse the splat-space forward transform from _spatial_crop_splat.py.
# (Same one the seeder uses, so this is consistent with the rest of the pipe.)


def _load_dataparser(dp_path: Path):
    d = json.loads(dp_path.read_text())
    T = np.asarray(d["transform"], dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    s = float(d["scale"])
    return R, t, s


def _metric_to_splat(positions_metric: np.ndarray,
                     R: np.ndarray, t: np.ndarray,
                     dp_scale: float, colmap_to_metric: float) -> np.ndarray:
    """Forward transform metric -> splat space.

    metric_pos / colmap_to_metric = colmap_pos
    splat_pos = dp_scale * (R @ colmap_pos + t)
    """
    colmap = positions_metric / colmap_to_metric
    splat = dp_scale * (colmap @ R.T + t[None, :])
    return splat.astype(np.float32)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_ply", type=Path)
    p.add_argument("--out", type=Path, default=None,
                   help="Output .ply. Default: <input>_cleaned_v3.ply")
    p.add_argument("--scale-pct", type=float, default=95.0)
    p.add_argument("--opacity-min", type=float, default=0.30)
    # AABB-from-mesh args
    p.add_argument("--scene-mesh", type=Path,
                   default=Path("output/mesh_v32_data3/mesh_v32_data3_openmvs.ply"))
    p.add_argument("--dataparser", type=Path,
                   default=Path("nerfstudio/dense/splatfacto/2026-06-07_203807/dataparser_transforms.json"))
    p.add_argument("--bounds-json", type=Path,
                   default=Path("colmap/dense/tof_bounds.json"))
    p.add_argument("--aabb-percentile", type=float, default=1.0,
                   help="Drop the bottom and top this %% of mesh vertices on "
                        "each axis before taking the AABB (default 1.0). "
                        "Catches a few stray OpenMVS outlier faces.")
    p.add_argument("--margin-m", type=float, default=0.10,
                   help="Metric margin (m) to expand the percentile AABB on "
                        "each axis (default 0.10 m).")
    p.add_argument("--no-aabb", action="store_true",
                   help="Disable AABB step. Only scale + opacity prune.")
    return p.parse_args()


def _scene_mesh_aabb_in_splat(scene_mesh_path: Path, dp_R, dp_t, dp_scale,
                               colmap_to_metric: float,
                               percentile: float, margin_m: float):
    """Read scene mesh, forward-transform vertices to splat-space, take a
    [pct, 100-pct] percentile AABB per axis, expand by metric margin."""
    import trimesh
    print(f"  scene mesh: {scene_mesh_path}")
    m = trimesh.load(str(scene_mesh_path), force="mesh", process=False)
    v_metric = np.asarray(m.vertices, dtype=np.float64)
    print(f"    {len(v_metric):,} vertices in metric world")
    print(f"    metric range: "
          f"x[{v_metric[:,0].min():+.3f},{v_metric[:,0].max():+.3f}]  "
          f"y[{v_metric[:,1].min():+.3f},{v_metric[:,1].max():+.3f}]  "
          f"z[{v_metric[:,2].min():+.3f},{v_metric[:,2].max():+.3f}]")
    v_splat = _metric_to_splat(v_metric, dp_R, dp_t, dp_scale, colmap_to_metric)
    print(f"    splat-space range: "
          f"x[{v_splat[:,0].min():+.3f},{v_splat[:,0].max():+.3f}]  "
          f"y[{v_splat[:,1].min():+.3f},{v_splat[:,1].max():+.3f}]  "
          f"z[{v_splat[:,2].min():+.3f},{v_splat[:,2].max():+.3f}]")
    # Percentile clip per axis
    lo = np.percentile(v_splat, percentile, axis=0)
    hi = np.percentile(v_splat, 100.0 - percentile, axis=0)
    print(f"  {percentile:.1f}-{100-percentile:.1f} percentile AABB: "
          f"x[{lo[0]:+.3f},{hi[0]:+.3f}]  "
          f"y[{lo[1]:+.3f},{hi[1]:+.3f}]  "
          f"z[{lo[2]:+.3f},{hi[2]:+.3f}]")
    # Margin in splat units (use same conversion as v2: dp_scale / colmap_to_metric)
    metric_to_splat = dp_scale / colmap_to_metric
    m_splat = margin_m * metric_to_splat
    lo -= m_splat
    hi += m_splat
    print(f"  AFTER +{margin_m*100:.0f} cm metric margin "
          f"(={m_splat:.3f} splat units): "
          f"x[{lo[0]:+.3f},{hi[0]:+.3f}]  "
          f"y[{lo[1]:+.3f},{hi[1]:+.3f}]  "
          f"z[{lo[2]:+.3f},{hi[2]:+.3f}]")
    return {"x_min": float(lo[0]), "x_max": float(hi[0]),
            "y_min": float(lo[1]), "y_max": float(hi[1]),
            "z_min": float(lo[2]), "z_max": float(hi[2])}


def main():
    args = _parse_args()
    in_path = args.input_ply.resolve()
    out_path = (args.out or in_path.with_name(in_path.stem + "_cleaned_v3.ply")).resolve()
    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        return 2

    print(f"=== aggressive_prune_v3 ===")
    print(f"  input  : {in_path}")
    print(f"  output : {out_path}")

    n, props, raw, rec_size, header = _read_ply(in_path)
    arr = np.frombuffer(raw, dtype=np.float32).reshape(n, len(props)).copy()
    name_to_idx = {p: i for i, p in enumerate(props)}
    print(f"  loaded {n:,} Gaussians, {len(props)} properties each")

    x = arr[:, name_to_idx["x"]]
    y = arr[:, name_to_idx["y"]]
    z = arr[:, name_to_idx["z"]]
    op_logit = arr[:, name_to_idx["opacity"]]
    op = 1.0 / (1.0 + np.exp(-op_logit))
    scale_keys = sorted([p for p in props if p.startswith("scale_")])
    scales = np.stack([arr[:, name_to_idx[k]] for k in scale_keys], axis=1)
    max_scale = np.exp(scales).max(axis=1)

    print(f"  input splat extent (splat units): "
          f"x[{x.min():+.2f},{x.max():+.2f}]  "
          f"y[{y.min():+.2f},{y.max():+.2f}]  "
          f"z[{z.min():+.2f},{z.max():+.2f}]")

    keep = np.ones(n, dtype=bool)

    # filter 1 — scale percentile
    scale_thresh = float(np.percentile(max_scale, args.scale_pct))
    scale_mask = max_scale <= scale_thresh
    dropped = int((~scale_mask).sum())
    print(f"\n  filter 1 — scale > p{args.scale_pct} ({scale_thresh:.4f}): "
          f"drop {dropped:,} Gaussians ({100*dropped/n:.1f}%)")
    keep &= scale_mask

    # filter 2 — opacity
    op_mask = op >= args.opacity_min
    dropped = int((~op_mask & keep).sum())
    print(f"  filter 2 — opacity < {args.opacity_min}: drop additional "
          f"{dropped:,} Gaussians")
    keep &= op_mask

    # filter 3 — mesh-derived splat-space AABB
    if not args.no_aabb:
        # Load dataparser + colmap_to_metric
        dp_R, dp_t, dp_scale = _load_dataparser(args.dataparser)
        bounds = json.loads(args.bounds_json.read_text())
        colmap_to_metric = 1.0 / float(bounds["scale_factor_da3_to_colmap"])
        print(f"\n  building mesh-derived AABB:")
        aabb = _scene_mesh_aabb_in_splat(
            args.scene_mesh, dp_R, dp_t, dp_scale, colmap_to_metric,
            args.aabb_percentile, args.margin_m)
        aabb_mask = ((x >= aabb["x_min"]) & (x <= aabb["x_max"]) &
                     (y >= aabb["y_min"]) & (y <= aabb["y_max"]) &
                     (z >= aabb["z_min"]) & (z <= aabb["z_max"]))
        dropped = int((~aabb_mask & keep).sum())
        n_pass_aabb = int(aabb_mask.sum())
        print(f"  filter 3 — mesh-derived AABB clip: "
              f"drop additional {dropped:,} Gaussians  "
              f"({n_pass_aabb:,}/{n:,} pass = {100*n_pass_aabb/n:.1f}%)")
        keep &= aabb_mask
    else:
        print(f"  filter 3 — AABB DISABLED (--no-aabb)")

    n_kept = int(keep.sum())
    pct_kept = 100 * n_kept / n
    print(f"\n  kept {n_kept:,}/{n:,} Gaussians ({pct_kept:.1f}%)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_ply(out_path, header, arr[keep])
    size_mb = out_path.stat().st_size / 1_048_576
    print(f"  saved: {out_path}  ({size_mb:.1f} MB)")
    print(f"\n=== Done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
