"""
aggressive_prune_v4 — DENSITY-based peripheral floater removal.

WHAT CHANGED vs v3
==================

v1/v2/v3 all relied on an axis-aligned bounding box (AABB) to clip
peripheral floaters. v3's AABB came from the OpenMVS scene mesh — but
that mesh missed the back wall, so v3 clipped real geometry along with
the floaters. Lesson: an AABB is the wrong tool when the "real" scene
and the "floaters" share the same coarse bounding extent. The fringe
isn't OUTSIDE the box; it's SPARSE inside it. The real scene is DENSE.

v4 replaces the AABB step with an Open3D density-outlier filter on the
Gaussian centroids. Two methods:

  • statistical (default)
        pcd.remove_statistical_outlier(nb_neighbors=K, std_ratio=R)
        Drops a point whose mean distance to its K nearest neighbours
        is above (overall_mean + R * overall_std). Lower R = more
        aggressive.

  • radius
        pcd.remove_radius_outlier(nb_points=N, radius=r)
        Drops a point with fewer than N neighbours inside `r`. Smaller
        radius or larger nb_points = more aggressive. The radius is in
        SPLAT-SPACE units; for V32 the metric_to_splat factor is 1.5227
        (so 0.10 splat-units ≈ 6.5 cm metric).

  • both = statistical, then radius (a point must survive both).
  • none = skip density (scale + opacity only).

The scale-percentile + stricter-opacity filters from v1/v2/v3 are kept
unchanged (defaults: p95 scale, opacity ≥ 0.30).

USAGE
=====
    python aggressive_prune_v4.py output/splat_v32_data3_noinit_pruned.ply \
        [--out output/splat_v32_data3_v4.ply] \
        [--method statistical|radius|both|none] \
        [--nb-neighbors 20] [--std-ratio 2.0] [--radius 0.10]

The script does NOT touch the input file.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Reuse v1's PLY reader/writer.
sys.path.insert(0, str(Path(__file__).parent))
from aggressive_prune import _read_ply, _write_ply                  # noqa: E402


# V32 splat-space conversion. See SCENE_SEGMENTER_NOTES / v3 for derivation:
#   metric_to_splat = dp_scale / colmap_to_metric.
METRIC_TO_SPLAT_V32 = 1.5227


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_ply", type=Path)
    p.add_argument("--out", type=Path, default=None,
                   help="Output .ply. Default: <input>_cleaned_v4.ply")
    p.add_argument("--scale-pct", type=float, default=95.0,
                   help="Drop Gaussians whose max-scale is above this "
                        "percentile (default 95).")
    p.add_argument("--opacity-min", type=float, default=0.30,
                   help="Drop Gaussians with sigmoid(opacity) < this "
                        "(default 0.30).")
    p.add_argument("--method", choices=("statistical", "radius", "both", "none"),
                   default="statistical",
                   help="Density-outlier method (default 'statistical'). "
                        "'both' = statistical then radius. 'none' = skip.")
    p.add_argument("--nb-neighbors", type=int, default=20,
                   help="K nearest neighbours for 'statistical', OR the "
                        "minimum neighbour count inside --radius for 'radius' "
                        "(default 20).")
    p.add_argument("--std-ratio", type=float, default=2.0,
                   help="Std-ratio for 'statistical'. Lower = more aggressive "
                        "(default 2.0).")
    p.add_argument("--radius", type=float, default=0.10,
                   help="Radius for 'radius' method, in splat-space units "
                        "(default 0.10).")
    return p.parse_args()


def _density_filter(positions: np.ndarray, method: str,
                    nb_neighbors: int, std_ratio: float, radius: float
                    ) -> np.ndarray:
    """Return a boolean mask (len = positions.shape[0]) of points that pass
    the chosen density filter."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions.astype(np.float64))
    n = positions.shape[0]
    mask = np.ones(n, dtype=bool)

    if method in ("statistical", "both"):
        _, ind = pcd.remove_statistical_outlier(
            nb_neighbors=int(nb_neighbors), std_ratio=float(std_ratio))
        sub = np.zeros(n, dtype=bool)
        sub[np.asarray(ind, dtype=np.int64)] = True
        dropped = int((~sub).sum())
        print(f"    statistical (k={nb_neighbors}, std_ratio={std_ratio}): "
              f"drop {dropped:,}/{n:,} ({100*dropped/n:.1f}%)")
        mask &= sub
        if method == "both":
            # Rebuild the PCD from survivors so the radius pass operates
            # on the same population the statistical pass kept.
            pcd2 = o3d.geometry.PointCloud()
            pcd2.points = o3d.utility.Vector3dVector(
                positions[mask].astype(np.float64))
            _, ind2 = pcd2.remove_radius_outlier(
                nb_points=int(nb_neighbors), radius=float(radius))
            sub2_local = np.zeros(int(mask.sum()), dtype=bool)
            sub2_local[np.asarray(ind2, dtype=np.int64)] = True
            sub2 = np.zeros(n, dtype=bool)
            idx_alive = np.where(mask)[0]
            sub2[idx_alive[sub2_local]] = True
            dropped2 = int((mask & ~sub2).sum())
            print(f"    radius (nb_points={nb_neighbors}, r={radius}): "
                  f"drop additional {dropped2:,}")
            mask &= sub2

    elif method == "radius":
        _, ind = pcd.remove_radius_outlier(
            nb_points=int(nb_neighbors), radius=float(radius))
        sub = np.zeros(n, dtype=bool)
        sub[np.asarray(ind, dtype=np.int64)] = True
        dropped = int((~sub).sum())
        print(f"    radius (nb_points={nb_neighbors}, r={radius}): "
              f"drop {dropped:,}/{n:,} ({100*dropped/n:.1f}%)")
        mask &= sub

    return mask


def main():
    args = _parse_args()
    in_path = args.input_ply.resolve()
    out_path = (args.out or in_path.with_name(in_path.stem + "_cleaned_v4.ply")).resolve()
    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        return 2

    print(f"=== aggressive_prune_v4 ===")
    print(f"  input  : {in_path}")
    print(f"  output : {out_path}")
    print(f"  method : {args.method}")
    # Effective metric size of the radius (V32 conversion).
    r_cm_metric = (args.radius / METRIC_TO_SPLAT_V32) * 100.0
    print(f"  radius {args.radius:.3f} splat ≈ {r_cm_metric:.2f} cm metric "
          f"(V32 metric_to_splat = {METRIC_TO_SPLAT_V32})")

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

    # ── filter 1 — scale percentile ──────────────────────────────────────
    scale_thresh = float(np.percentile(max_scale, args.scale_pct))
    scale_mask = max_scale <= scale_thresh
    dropped = int((~scale_mask).sum())
    remaining = int((keep & scale_mask).sum())
    print(f"\n  filter 1 — scale > p{args.scale_pct} ({scale_thresh:.4f}): "
          f"drop {dropped:,} Gaussians ({100*dropped/n:.1f}%); "
          f"{remaining:,} remain")
    keep &= scale_mask

    # ── filter 2 — opacity threshold ─────────────────────────────────────
    op_mask = op >= args.opacity_min
    dropped = int((~op_mask & keep).sum())
    keep &= op_mask
    remaining = int(keep.sum())
    print(f"  filter 2 — opacity < {args.opacity_min}: drop additional "
          f"{dropped:,} Gaussians; {remaining:,} remain")

    # ── filter 3 — DENSITY outlier removal ───────────────────────────────
    if args.method == "none":
        print(f"  filter 3 — density DISABLED (method=none)")
    else:
        alive_idx = np.where(keep)[0]
        positions = np.stack([x[alive_idx], y[alive_idx], z[alive_idx]], axis=1)
        print(f"\n  filter 3 — density ({args.method}) on {len(alive_idx):,} "
              f"surviving Gaussians:")
        dens_local_mask = _density_filter(
            positions, args.method,
            args.nb_neighbors, args.std_ratio, args.radius)
        density_mask = np.zeros(n, dtype=bool)
        density_mask[alive_idx[dens_local_mask]] = True
        before = int(keep.sum())
        keep &= density_mask
        after = int(keep.sum())
        print(f"    density step total: {before:,} -> {after:,} "
              f"(drop {before - after:,})")

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
