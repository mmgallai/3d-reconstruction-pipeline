"""
Subsample a dense PLY point cloud to a target number of points.
Uses random uniform sampling. Preserves all vertex properties.

Usage:
    python subsample_ply.py --input colmap/dense/fused.ply
                            --output colmap/dense/fused_sub.ply
                            --target 500000
"""
import argparse
import numpy as np
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--target", type=int, default=500000)
    p.add_argument("--sigma-clip", type=float, default=3.0,
                   help="Remove points beyond this many σ from the mean in any axis (default: 3.0, 0=off).")
    return p.parse_args()


def main():
    args = parse_args()
    src = Path(args.input)
    dst = Path(args.output)

    print(f"Reading {src} ...")
    with open(src, "rb") as f:
        header_lines = []
        while True:
            line = f.readline().decode("ascii", errors="replace")
            header_lines.append(line)
            if line.strip() == "end_header":
                break
        data_start = f.tell()
        raw = f.read()

    # Parse header
    n_total = 0
    properties = []
    for line in header_lines:
        if line.strip().startswith("element vertex"):
            n_total = int(line.strip().split()[-1])
        elif line.strip().startswith("property"):
            properties.append(line.strip())

    # Build numpy dtype from PLY property types
    ply_to_np = {
        "float": np.float32, "float32": np.float32,
        "double": np.float64, "float64": np.float64,
        "uchar": np.uint8,  "uint8": np.uint8,
        "char":  np.int8,   "int8":  np.int8,
        "short": np.int16,  "int16": np.int16,
        "ushort": np.uint16,"uint16": np.uint16,
        "int":   np.int32,  "int32": np.int32,
        "uint":  np.uint32, "uint32": np.uint32,
    }
    dtype_fields = []
    for prop in properties:
        parts = prop.split()  # e.g. ["property", "float", "x"]
        ptype, pname = parts[1], parts[2]
        dtype_fields.append((pname, ply_to_np.get(ptype, np.float32)))

    dtype = np.dtype(dtype_fields)
    print(f"  {n_total:,} points, dtype: {dtype}")

    data = np.frombuffer(raw[:n_total * dtype.itemsize], dtype=dtype).copy()

    # Spatial outlier removal
    if args.sigma_clip > 0 and all(f in data.dtype.names for f in ("x", "y", "z")):
        x = data["x"].astype(np.float32)
        y = data["y"].astype(np.float32)
        z = data["z"].astype(np.float32)
        mask = np.ones(len(data), dtype=bool)
        for arr in (x, y, z):
            mean, std = arr.mean(), arr.std()
            mask &= np.abs(arr - mean) <= args.sigma_clip * std
        before = len(data)
        data = data[mask]
        removed = before - len(data)
        print(f"  Outlier clip ({args.sigma_clip}σ): removed {removed:,} points ({100*removed/before:.1f}%),  {len(data):,} remain")
        n_total = len(data)

    # Subsample
    target = min(args.target, n_total)
    if target >= n_total:
        print(f"  Already <= {target:,} points — copying as-is.")
        idx = np.arange(n_total)
    else:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_total, size=target, replace=False)
        idx.sort()
        print(f"  Subsampling {n_total:,} → {target:,} points ({100*target/n_total:.1f}%)")

    sampled = data[idx]

    # Rebuild header with new count
    new_header = ""
    for line in header_lines:
        if line.strip().startswith("element vertex"):
            new_header += f"element vertex {len(idx)}\n"
        else:
            new_header += line

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        f.write(new_header.encode("ascii"))
        f.write(sampled.tobytes("C"))

    size_mb = dst.stat().st_size / 1_048_576
    print(f"  Saved: {dst}  ({size_mb:.0f} MB, {len(idx):,} points)")


if __name__ == "__main__":
    main()
