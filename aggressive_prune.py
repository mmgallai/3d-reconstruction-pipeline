"""
Aggressive splat pruner — strips reflection blobs + sparse haze from a
trained Gaussian Splat .ply without retraining.

Adds three filters beyond the existing prune_splat.py (which only does
opacity + connected-component cleanup):

  1. **Scale percentile** — drops Gaussians whose max-scale axis is
     above the chosen percentile (default 95). Floaters and reflection
     blobs are usually physically larger than surface-anchored ones;
     this clips that tail.

  2. **Stronger opacity threshold** — defaults to 0.30 (vs prune_splat's
     0.15). Reflection blobs survive 0.15 because they DID minimise loss
     and have high opacity in their local neighbourhood — but the
     floaters around them are typically 0.15-0.30, so a stricter cut
     catches the haze.

  3. **Scene AABB clip** (optional) — drops Gaussians outside an axis-
     aligned bounding box read from `colmap/dense/tof_bounds.json`.
     Eliminates Gaussians that drifted outside the captured scene.

Usage:
    python aggressive_prune.py output/splat_v32_noinit_pruned.ply \
        [--out output/splat_v32_clean.ply] \
        [--scale-pct 95] [--opacity-min 0.30] [--use-aabb]

The script does NOT touch the input file. A fresh .ply is written.
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_ply", type=Path)
    p.add_argument("--out", type=Path, default=None,
                   help="Output .ply. Default: <input>_clean.ply")
    p.add_argument("--scale-pct", type=float, default=95.0,
                   help="Drop Gaussians whose max-scale is above this "
                        "percentile of the population (default 95)")
    p.add_argument("--opacity-min", type=float, default=0.30,
                   help="Drop Gaussians with sigmoid(opacity) < this "
                        "(default 0.30; pipeline default for the standard "
                        "prune is 0.15)")
    p.add_argument("--use-aabb", action="store_true",
                   help="Clip to scene AABB read from "
                        "colmap/dense/tof_bounds.json (with 20%% slack)")
    p.add_argument("--bounds-json", type=Path,
                   default=Path("colmap/dense/tof_bounds.json"))
    p.add_argument("--aabb-slack", type=float, default=0.20,
                   help="Fractional slack on each axis when applying AABB "
                        "clip (default 0.20 = 20%% bigger)")
    return p.parse_args()


def _read_ply(path: Path):
    """Read binary-little-endian PLY with float32 vertex properties.
    Returns (n_vertices, list_of_property_names, raw_data_bytes,
             property_offsets, vertex_record_size, header_text)."""
    data = path.read_bytes()
    end_marker = b"end_header\n"
    header_end = data.find(end_marker) + len(end_marker)
    header = data[:header_end].decode("ascii", errors="replace")
    lines = header.splitlines()

    n = 0
    props = []
    in_vertex = False
    for line in lines:
        if line.startswith("element vertex "):
            n = int(line.split()[2])
            in_vertex = True
        elif line.startswith("element "):
            in_vertex = False
        elif in_vertex and line.startswith("property "):
            parts = line.split()
            if parts[1] == "float":
                props.append(parts[2])
            elif parts[1] == "uchar":
                props.append((parts[2], "uchar"))
            else:
                raise RuntimeError(f"Unsupported property type: {line}")

    # We only support all-float32 vertex records (matches splatfacto output).
    if any(isinstance(p, tuple) for p in props):
        raise RuntimeError("This pruner only supports float32 splat .plys")

    record_size = 4 * len(props)
    expected = header_end + n * record_size
    if len(data) < expected:
        raise RuntimeError(f"PLY truncated: expected {expected} bytes, got {len(data)}")

    raw = data[header_end:header_end + n * record_size]
    return n, props, raw, record_size, header


def _write_ply(out_path: Path, header: str, kept_rows: np.ndarray):
    """kept_rows is a (n_kept, n_props) float32 array."""
    n_kept = kept_rows.shape[0]
    # Replace the vertex count in the header
    new_header = []
    for line in header.splitlines():
        if line.startswith("element vertex "):
            new_header.append(f"element vertex {n_kept}")
        else:
            new_header.append(line)
    new_header_bytes = ("\n".join(new_header) + "\n").encode("ascii")
    out_path.write_bytes(new_header_bytes + kept_rows.astype(np.float32).tobytes())


def main():
    args = _parse_args()
    in_path = args.input_ply.resolve()
    out_path = (args.out or in_path.with_name(in_path.stem + "_clean.ply")).resolve()
    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        return 2

    print(f"=== aggressive_prune ===")
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

    print(f"  current stats:")
    print(f"    opacity median {np.median(op):.3f}  min {op.min():.3f}  max {op.max():.3f}")
    print(f"    max-scale median {np.median(max_scale):.4f}  "
          f"p95 {np.percentile(max_scale, 95):.4f}  "
          f"max {max_scale.max():.4f}")
    print(f"    BB: x[{x.min():.2f},{x.max():.2f}]  "
          f"y[{y.min():.2f},{y.max():.2f}]  "
          f"z[{z.min():.2f},{z.max():.2f}]")

    keep = np.ones(n, dtype=bool)

    # ── Filter 1: scale percentile (the big lever for reflection blobs) ───
    scale_thresh = float(np.percentile(max_scale, args.scale_pct))
    scale_mask = max_scale <= scale_thresh
    dropped = int((~scale_mask).sum())
    print(f"  filter 1 — scale > p{args.scale_pct} ({scale_thresh:.4f}): "
          f"drop {dropped:,} Gaussians ({100*dropped/n:.1f}%)")
    keep &= scale_mask

    # ── Filter 2: stricter opacity threshold ─────────────────────────────
    op_mask = op >= args.opacity_min
    dropped = int((~op_mask & keep).sum())
    print(f"  filter 2 — opacity < {args.opacity_min}: drop additional "
          f"{dropped:,} Gaussians")
    keep &= op_mask

    # ── Filter 3 (optional): scene AABB clip ─────────────────────────────
    if args.use_aabb:
        bp = args.bounds_json.resolve()
        if not bp.exists():
            print(f"  WARN: --use-aabb set but {bp} not found; skipping")
        else:
            b = json.loads(bp.read_text())
            slack = args.aabb_slack
            sx = (b["x_max"] - b["x_min"]) * slack
            sy = (b["y_max"] - b["y_min"]) * slack
            sz = (b["z_max"] - b["z_min"]) * slack
            x_lo, x_hi = b["x_min"] - sx, b["x_max"] + sx
            y_lo, y_hi = b["y_min"] - sy, b["y_max"] + sy
            z_lo, z_hi = b["z_min"] - sz, b["z_max"] + sz
            aabb_mask = ((x >= x_lo) & (x <= x_hi) &
                         (y >= y_lo) & (y <= y_hi) &
                         (z >= z_lo) & (z <= z_hi))
            dropped = int((~aabb_mask & keep).sum())
            print(f"  filter 3 — AABB clip ±{int(slack*100)}% slack: "
                  f"drop additional {dropped:,} Gaussians")
            keep &= aabb_mask

    kept = arr[keep]
    print(f"\n  kept {keep.sum():,} / {n:,} Gaussians ({100*keep.sum()/n:.1f}%)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_ply(out_path, header, kept)
    size_mb = out_path.stat().st_size / 1_048_576
    print(f"  saved: {out_path}  ({size_mb:.1f} MB)")
    print(f"\n=== Done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
