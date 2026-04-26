"""
Prune low-opacity (ghost/floater) Gaussians from a splat.ply.
Removes Gaussians whose opacity (after sigmoid) is below a threshold.
Also removes extreme spatial outliers (>5σ from mean in any axis).

Usage:
    python prune_splat.py [--threshold 0.15] [--no-outlier-prune]
"""

import argparse
import struct
import numpy as np
from pathlib import Path

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  default="output/splat.ply")
    p.add_argument("--output", default="output/splat_pruned.ply")
    p.add_argument("--threshold", type=float, default=0.15,
                   help="Minimum sigmoid-opacity to keep (default 0.15)")
    p.add_argument("--sigma-outlier", type=float, default=5.0,
                   help="Remove Gaussians >N sigma from mean position (default 5.0, 0=disable)")
    return p.parse_args()

def main():
    args = parse_args()
    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading {src} ...")
    with open(src, "rb") as f:
        header_bytes = b""
        header_lines = []
        while True:
            line = f.readline()
            header_bytes += line
            decoded = line.decode("ascii", errors="replace").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break
        data_start = f.tell()
        raw = f.read()

    # Parse property list
    n_total = 0
    properties = []
    for line in header_lines:
        if line.startswith("element vertex"):
            n_total = int(line.split()[-1])
        elif line.startswith("property float"):
            properties.append(line.split()[-1])

    n_props = len(properties)
    prop_idx = {name: i for i, name in enumerate(properties)}
    print(f"  {n_total:,} Gaussians, {n_props} properties each")

    dtype = np.dtype([(f"f{i}", np.float32) for i in range(n_props)])
    data = np.frombuffer(raw, dtype=dtype).copy()
    cols = np.stack([data[f"f{i}"] for i in range(n_props)], axis=1)  # (N, P)

    mask = np.ones(n_total, dtype=bool)

    # ── Opacity filter ────────────────────────────────────────────────────────
    if "opacity" in prop_idx:
        raw_op = cols[:, prop_idx["opacity"]]
        op = sigmoid(raw_op)
        op_mask = op >= args.threshold
        removed_op = np.sum(~op_mask)
        mask &= op_mask
        print(f"  Opacity < {args.threshold}: removing {removed_op:,} ({100*removed_op/n_total:.1f}%)")

    # ── Spatial outlier filter ────────────────────────────────────────────────
    if args.sigma_outlier > 0:
        for ax_name in ("x", "y", "z"):
            if ax_name not in prop_idx:
                continue
            ax = cols[:, prop_idx[ax_name]]
            mu, sd = ax[mask].mean(), ax[mask].std()
            out_mask = np.abs(ax - mu) <= args.sigma_outlier * sd
            removed_ax = np.sum(mask & ~out_mask)
            if removed_ax > 0:
                print(f"  {ax_name} outliers (>{args.sigma_outlier}σ): removing {removed_ax:,}")
            mask &= out_mask

    kept = int(mask.sum())
    print(f"  Keeping {kept:,} / {n_total:,} Gaussians ({100*kept/n_total:.1f}%)")

    # ── Write output PLY ──────────────────────────────────────────────────────
    # Rebuild header with updated vertex count
    new_header_lines = []
    for line in header_lines:
        if line.startswith("element vertex"):
            new_header_lines.append(f"element vertex {kept}")
        else:
            new_header_lines.append(line)
    new_header = "\n".join(new_header_lines) + "\n"

    pruned_data = cols[mask].astype(np.float32)

    print(f"  Writing {dst} ...")
    with open(dst, "wb") as f:
        f.write(new_header.encode("ascii"))
        f.write(pruned_data.tobytes())

    size_mb = dst.stat().st_size / 1_048_576
    print(f"  Done. {size_mb:.0f} MB  ({kept:,} Gaussians)")
    print(f"\nPruned splat saved to: {dst}")

if __name__ == "__main__":
    main()
