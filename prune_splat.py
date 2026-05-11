"""
Prune low-opacity (ghost/floater) Gaussians from a splat.ply.

Three-stage filter:
  1. Opacity threshold      — drop Gaussians with sigmoid-opacity below `--threshold`
  2. Spatial σ-outliers     — drop Gaussians > N σ from mean position in any axis
  3. Connected-component    — drop Gaussians not in the largest spatial blob
                              (catches isolated floater clusters that survive 1+2)

Usage:
    python prune_splat.py --input splat.ply --output splat_pruned.ply
"""

import argparse
import struct
from collections import deque
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
    p.add_argument("--cc-voxel", type=float, default=0.10,
                   help="Voxel size (world units) for connected-component grouping. "
                        "Smaller = stricter (more components). 0=disable. Default 0.10.")
    p.add_argument("--cc-min-frac", type=float, default=0.005,
                   help="Drop components smaller than this fraction of the largest "
                        "(default 0.005 = 0.5%%; 0=keep only the largest component).")
    return p.parse_args()


def connected_component_filter(xyz, voxel_size, min_frac=0.005):
    """
    Spatial connected-component pruning.

    Quantize positions to a voxel grid, build a sparse occupancy set, then BFS
    through 26-neighbours to find connected blobs.  Returns a boolean mask
    keeping all Gaussians whose voxel belongs to a component with size
    >= min_frac × largest_component_size.

    voxel_size  – grid resolution in world units (0.10 ≈ 10 cm for our scenes)
    min_frac    – minimum component size as a fraction of the largest blob
                  (0.005 keeps anything ≥ 0.5 % of the main scene).  Set to
                  1.0 to keep ONLY the largest component (most aggressive).
    """
    if voxel_size <= 0:
        return np.ones(len(xyz), dtype=bool)

    # ── 1. Quantise to voxel keys ─────────────────────────────────────────────
    keys = np.floor(xyz / voxel_size).astype(np.int32)            # (N,3)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)      # (V,3), (N,)
    n_vox = len(uniq)

    # Build (key → voxel_index) hash for neighbour lookup
    key_to_idx = {tuple(k): i for i, k in enumerate(uniq)}

    # ── 2. BFS to label connected components ──────────────────────────────────
    label = np.full(n_vox, -1, dtype=np.int32)
    n_components = 0
    component_size = []

    # 26-neighbour offsets (excluding 0,0,0)
    offsets = [(dx, dy, dz)
               for dx in (-1, 0, 1)
               for dy in (-1, 0, 1)
               for dz in (-1, 0, 1)
               if (dx, dy, dz) != (0, 0, 0)]

    for v_idx in range(n_vox):
        if label[v_idx] != -1:
            continue
        # New component — BFS
        label[v_idx] = n_components
        size = 0
        q = deque([v_idx])
        while q:
            i = q.popleft()
            size += 1
            kx, ky, kz = uniq[i]
            for dx, dy, dz in offsets:
                nb = (int(kx) + dx, int(ky) + dy, int(kz) + dz)
                j = key_to_idx.get(nb)
                if j is not None and label[j] == -1:
                    label[j] = n_components
                    q.append(j)
        component_size.append(size)
        n_components += 1

    component_size = np.array(component_size)
    largest = component_size.max()

    # ── 3. Keep voxels in components ≥ min_frac of largest ────────────────────
    threshold = max(1, int(round(min_frac * largest)))
    keep_components = np.where(component_size >= threshold)[0]

    print(f"  CC voxel grid: {n_vox:,} occupied voxels, "
          f"{n_components} components, largest = {largest:,} voxels")
    print(f"  Keeping {len(keep_components)} components "
          f"(size ≥ {threshold:,} voxels, threshold = {min_frac*100:.2f}% of largest)")

    keep_set = set(keep_components.tolist())
    voxel_kept = np.array([lbl in keep_set for lbl in label], dtype=bool)
    return voxel_kept[inv]

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

    # ── Connected-component filter (spatial, after opacity + σ-outlier) ───────
    # Catches isolated floater clusters that survive both prior filters.
    # Operates only on Gaussians still kept by mask above.
    if args.cc_voxel > 0 and all(a in prop_idx for a in ("x", "y", "z")):
        kept_idx = np.where(mask)[0]
        xyz_kept = np.stack([
            cols[kept_idx, prop_idx["x"]],
            cols[kept_idx, prop_idx["y"]],
            cols[kept_idx, prop_idx["z"]],
        ], axis=1)
        cc_mask_local = connected_component_filter(
            xyz_kept, args.cc_voxel, args.cc_min_frac
        )
        removed_cc = int(np.sum(~cc_mask_local))
        if removed_cc > 0:
            print(f"  CC filter: removing {removed_cc:,} Gaussians in small/disconnected blobs")
        # Re-project into the global mask
        mask[kept_idx] = cc_mask_local

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
