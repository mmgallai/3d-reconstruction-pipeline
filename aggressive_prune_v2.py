"""
aggressive_prune_v2 — fixes the coordinate-frame bug in v1's AABB clip and
auto-derives a SCENE-TIGHT box in splat-space.

WHAT THIS VERSION CHANGES (vs aggressive_prune.py v1)
=====================================================

v1 read AABB extents from `colmap/dense/tof_bounds.json` (which are in
COLMAP units, ~9× larger than splat-space metres on V32) and compared them
DIRECTLY against the PLY's positions (which are in nerfstudio splat-space
post-dataparser transform). On V32 that meant the effective box on x and y
was ~10× too wide, so the AABB filter was nearly a no-op on those axes
(only the z range happened to overlap enough to drop ~22% of Gaussians,
mostly as an arbitrary half-space cut rather than a desk box). Result:
the leafy surrounding-room fringe was inside the broken box and survived.

v2 fixes this by working entirely in **splat-space**:

  • **`--auto-aabb`** (default ON). Reads the per-object splat PLYs from the
    `--segmenter-dir` (default `output/segmented_<scene>_v9a_fp_v2_splat_v6_cleaned`),
    takes their union AABB, and expands by `--margin-m` (default 0.30 m
    metric) on each axis so the desk and reasonable surroundings stay.
    This is the same coordinate frame the input PLY lives in -- no
    conversion bugs possible.

  • **`--aabb-json <path>`**. Provide an explicit splat-space AABB JSON
    instead. Schema:
        {"x_min": ..., "x_max": ..., "y_min": ..., "y_max": ...,
         "z_min": ..., "z_max": ...}

  • **`--aabb-x-min/...`** explicit CLI overrides if you just want to type
    six numbers.

  • Scale + opacity defaults UNCHANGED from v1 (95th percentile + 0.30)
    so per-object interior quality is not compromised. Only the AABB
    step is the difference.

USAGE
=====
    python aggressive_prune_v2.py output/splat_v32_data3_noinit.ply \
        [--out output/splat_v32_data3_cleaned_v2.ply] \
        [--scale-pct 95] [--opacity-min 0.30] \
        [--auto-aabb] [--margin-m 0.30]

Or with an explicit box:
    python aggressive_prune_v2.py <in.ply> --aabb-json desk_box.json
    python aggressive_prune_v2.py <in.ply> \
        --aabb-x-min -1.1 --aabb-x-max 0.8 \
        --aabb-y-min  0.55 --aabb-y-max 1.2 \
        --aabb-z-min -0.65 --aabb-z-max 0.15
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Reuse v1's PLY reader/writer to keep behaviour identical on the IO side.
sys.path.insert(0, str(Path(__file__).parent))
from aggressive_prune import _read_ply, _write_ply  # noqa: E402


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_ply", type=Path)
    p.add_argument("--out", type=Path, default=None,
                   help="Output .ply. Default: <input>_cleaned_v2.ply")
    p.add_argument("--scale-pct", type=float, default=95.0,
                   help="Drop Gaussians whose max-scale axis exceeds this "
                        "percentile of the population (default 95).")
    p.add_argument("--opacity-min", type=float, default=0.30,
                   help="Drop Gaussians with sigmoid(opacity) < this "
                        "(default 0.30).")
    # ── AABB sources (mutually compatible; precedence: explicit > json > auto) ─
    p.add_argument("--auto-aabb", action="store_true", default=True,
                   help="(Default ON) auto-derive a splat-space AABB from the "
                        "segmenter's per-object PLYs + a metric margin.")
    p.add_argument("--no-aabb", action="store_true",
                   help="Disable the AABB filter entirely. Only scale + "
                        "opacity prune will run.")
    p.add_argument("--segmenter-dir", type=Path,
                   default=Path("output/segmented_v32_data3_v9a_fp_v2_splat_v6_cleaned"),
                   help="Directory of per-object splat PLYs whose union AABB "
                        "we'll expand to derive the keep-box. Used only when "
                        "--auto-aabb is on AND --aabb-json is not given.")
    p.add_argument("--margin-m", type=float, default=0.30,
                   help="Metric margin to expand the auto-derived AABB on "
                        "each axis (default 0.30 m). Larger = keeps more "
                        "surrounding context.")
    p.add_argument("--metric-to-splat", type=float, default=1.5227,
                   help="Conversion factor metric-metres -> splat-space units "
                        "for the V32 pipeline. dp_scale × scale_factor_da3_to_colmap "
                        "= 0.17306 × 8.7986 = 1.5227.")
    p.add_argument("--aabb-json", type=Path, default=None,
                   help="Explicit splat-space AABB JSON "
                        "(keys: x_min, x_max, y_min, y_max, z_min, z_max). "
                        "Overrides --auto-aabb.")
    p.add_argument("--aabb-x-min", type=float, default=None)
    p.add_argument("--aabb-x-max", type=float, default=None)
    p.add_argument("--aabb-y-min", type=float, default=None)
    p.add_argument("--aabb-y-max", type=float, default=None)
    p.add_argument("--aabb-z-min", type=float, default=None)
    p.add_argument("--aabb-z-max", type=float, default=None)
    return p.parse_args()


def _aabb_from_segmenter(segmenter_dir: Path) -> dict | None:
    """Union AABB (in splat-space) of every `*_splat.ply` in the segmenter
    directory. Returns None if no PLYs found."""
    from plyfile import PlyData
    plys = sorted(segmenter_dir.glob("*_splat.ply"))
    if not plys:
        return None
    xs, ys, zs = [], [], []
    print(f"  auto-AABB: scanning {len(plys)} per-object splats in "
          f"{segmenter_dir.name}/")
    for p in plys:
        v = PlyData.read(str(p))["vertex"]
        x = np.asarray(v["x"]); y = np.asarray(v["y"]); z = np.asarray(v["z"])
        xs.append((x.min(), x.max())); ys.append((y.min(), y.max()))
        zs.append((z.min(), z.max()))
        print(f"    {p.name:50s}  x[{x.min():+.2f},{x.max():+.2f}]  "
              f"y[{y.min():+.2f},{y.max():+.2f}]  "
              f"z[{z.min():+.2f},{z.max():+.2f}]")
    return {
        "x_min": float(min(a for a, _ in xs)),
        "x_max": float(max(b for _, b in xs)),
        "y_min": float(min(a for a, _ in ys)),
        "y_max": float(max(b for _, b in ys)),
        "z_min": float(min(a for a, _ in zs)),
        "z_max": float(max(b for _, b in zs)),
    }


def _expand_aabb_metric(aabb: dict, margin_m: float, metric_to_splat: float) -> dict:
    """Expand each axis by `margin_m` metres (converted to splat units)."""
    m = margin_m * metric_to_splat
    return {
        "x_min": aabb["x_min"] - m, "x_max": aabb["x_max"] + m,
        "y_min": aabb["y_min"] - m, "y_max": aabb["y_max"] + m,
        "z_min": aabb["z_min"] - m, "z_max": aabb["z_max"] + m,
    }


def _resolve_aabb(args) -> dict | None:
    """Pick a splat-space AABB from the available sources.
    Precedence: explicit 6-arg > --aabb-json > --auto-aabb. Returns None if
    none are usable or --no-aabb."""
    if args.no_aabb:
        return None
    explicit = (args.aabb_x_min, args.aabb_x_max,
                args.aabb_y_min, args.aabb_y_max,
                args.aabb_z_min, args.aabb_z_max)
    if all(v is not None for v in explicit):
        return {"x_min": args.aabb_x_min, "x_max": args.aabb_x_max,
                "y_min": args.aabb_y_min, "y_max": args.aabb_y_max,
                "z_min": args.aabb_z_min, "z_max": args.aabb_z_max}
    if args.aabb_json is not None:
        if not args.aabb_json.is_file():
            print(f"  WARN --aabb-json {args.aabb_json} missing", file=sys.stderr)
        else:
            return json.loads(args.aabb_json.read_text())
    if args.auto_aabb:
        if not args.segmenter_dir.is_dir():
            print(f"  WARN --segmenter-dir {args.segmenter_dir} missing; "
                  f"skipping AABB filter", file=sys.stderr)
            return None
        aabb = _aabb_from_segmenter(args.segmenter_dir)
        if aabb is None:
            return None
        print(f"  raw object-union AABB (splat units):  "
              f"x[{aabb['x_min']:+.2f},{aabb['x_max']:+.2f}]  "
              f"y[{aabb['y_min']:+.2f},{aabb['y_max']:+.2f}]  "
              f"z[{aabb['z_min']:+.2f},{aabb['z_max']:+.2f}]")
        aabb = _expand_aabb_metric(aabb, args.margin_m, args.metric_to_splat)
        print(f"  AFTER +{args.margin_m*100:.0f} cm metric margin "
              f"(={args.margin_m*args.metric_to_splat:.3f} splat units):  "
              f"x[{aabb['x_min']:+.2f},{aabb['x_max']:+.2f}]  "
              f"y[{aabb['y_min']:+.2f},{aabb['y_max']:+.2f}]  "
              f"z[{aabb['z_min']:+.2f},{aabb['z_max']:+.2f}]")
        return aabb
    return None


def main():
    args = _parse_args()
    in_path = args.input_ply.resolve()
    out_path = (args.out or in_path.with_name(in_path.stem + "_cleaned_v2.ply")).resolve()
    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        return 2

    print(f"=== aggressive_prune_v2 ===")
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

    # ── Filter 1: scale percentile ───────────────────────────────────────────
    scale_thresh = float(np.percentile(max_scale, args.scale_pct))
    scale_mask = max_scale <= scale_thresh
    dropped = int((~scale_mask).sum())
    print(f"\n  filter 1 — scale > p{args.scale_pct} ({scale_thresh:.4f}): "
          f"drop {dropped:,} Gaussians ({100*dropped/n:.1f}%)")
    keep &= scale_mask

    # ── Filter 2: opacity threshold ──────────────────────────────────────────
    op_mask = op >= args.opacity_min
    dropped = int((~op_mask & keep).sum())
    print(f"  filter 2 — opacity < {args.opacity_min}: drop additional "
          f"{dropped:,} Gaussians")
    keep &= op_mask

    # ── Filter 3: splat-space AABB clip (THE v2 fix) ─────────────────────────
    aabb = _resolve_aabb(args)
    if aabb is not None:
        aabb_mask = ((x >= aabb["x_min"]) & (x <= aabb["x_max"]) &
                     (y >= aabb["y_min"]) & (y <= aabb["y_max"]) &
                     (z >= aabb["z_min"]) & (z <= aabb["z_max"]))
        dropped = int((~aabb_mask & keep).sum())
        n_kept_aabb = int(aabb_mask.sum())
        pct_aabb_kept = 100 * n_kept_aabb / n
        print(f"  filter 3 — splat-space AABB clip: "
              f"drop additional {dropped:,} Gaussians  "
              f"({n_kept_aabb:,}/{n:,} pass = {pct_aabb_kept:.1f}%)")
        keep &= aabb_mask
    else:
        print(f"  filter 3 — AABB DISABLED (--no-aabb, or no source given)")

    n_kept = int(keep.sum())
    pct_kept = 100 * n_kept / n
    print(f"\n  kept {n_kept:,}/{n:,} Gaussians ({pct_kept:.1f}%)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept_rows = arr[keep]
    _write_ply(out_path, header, kept_rows)
    size_mb = out_path.stat().st_size / 1_048_576
    print(f"  saved: {out_path}  ({size_mb:.1f} MB)")
    print(f"\n=== Done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
